#include "heat_parabolic_cuda.h"

// define physical constants
const float kappa = 1.0;



__global__ void heat_1d(float * T, uint n, float dt, float * x, uint Lx, uint sx, uint m_b){

	// Find index from threadId
	uint i = blockIdx.x * sx + threadIdx.x;

	if(i > 1 and i < Lx -1){

		// Load stencil
		float T9 = load_T(T, n, i - 1, 0, 0, Lx, 1, 1);
		float T0 = load_T(T, n, i + 0, 0, 0, Lx, 1, 1);
		float T1 = load_T(T, n, i + 1, 0, 0, Lx, 1, 1);

		// Load position grid
		float x9 = load_x(x, i - 1);
		float x0 = load_x(x, i + 0);
		float x1 = load_x(x, i + 1);

		// find first derivatives
		//		float Dx_T0 = D_F_1D(T1, T0, x1, x0);
		//		float Dx_T9 = D_F_1D(T0, T9, x0, x9);
		//
		//		// find second derivatives
		//		float DDx_T0 = D_F_1D(Dx_T0, Dx_T9, (x1 + x0) / 2.0, (x0 + x9) / 2.0);
		float DDx_T0 = (T9 - 2 * T0 + T1) / ((x1 - x0) * (x0 - x9));

		// compute time-update
		float Tn = T0 + dt * (kappa * DDx_T0);

		//		if(i == 128){
		//			printf("n = %04d, i=%04d, T=%e\n",n,i, DDx_T0);
		//		}
		// update global memory
		T[(n + 1) * Lx + i] = Tn;
	} else {	// boundary condition
		T[(n + 1) * Lx + i] = 0;
	}

	return;

}

__global__ void heat_1d_cuda(dim3 blocks, dim3 threads, float * T, float dt, float * x, uint Lt, uint Lx, uint sx,  uint m_b){

	// main time-marching loop
	for(uint n = 0; n < Lt - 1; n++){

		heat_1d<<<blocks, threads>>>(T, n, dt, x, Lx, sx, m_b);

		cudaDeviceSynchronize();

	}

	return;

}


void heat_1d_cpu(float * T, float dt, float * x, uint Lt, uint Lx, uint m_f, uint m_b){

	for(uint n = 0; n < Lt-1; n++){

		for(uint i = 0; i < Lx; i++){

			if(i != 0 and i != Lx - 1){

				// Load stencil
				float T9 = T[n * Lx + (i - 1)];
				float T0 = T[n * Lx + (i + 0)];
				float T1 = T[n * Lx + (i + 1)];

				// Load position grid
				float x9 = x[i - 1];
				float x0 = x[i + 0];
				float x1 = x[i + 1];;

				// compute second derivative
				float DDx_T0 = (T9 - 2 * T0 + T1) / ((x1 - x0) * (x0 - x9));

				// compute time-update
				float Tn = T0 + dt * (kappa * DDx_T0);

				// update global memory
				T[(n + 1) * Lx + i] = Tn;
			} else {	// boundary condition
				T[(n + 1) * Lx + i] = 0;
			}

		}

	}

	return;

}

int main(void)
{

	// Set up device
	int dev = 0;
	CHECK(cudaSetDevice(dev));

	// Print device and precision
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, 0));
	//	print_device_properties(prop);

	// specify the order of the differentiation in each direction
	uint m_f = 1;
	uint m_b = 1;
	uint m = m_b + m_f;

	// size of strides
	uint sx = 1024;
	uint sy = 1;
	uint sz = 1;

	// number of strides
	uint Nx = 2048;
	uint Ny = 1;
	uint Nz = 1;

	// Size of domain in gridpoints (including boundary cells)
	uint Lt = 32;
	uint Lx = Nx * sx;
	uint Ly = Ny * sy;
	uint Lz = Nz * sz;
	uint L = Lx * Ly * Lz * Lt;

	// Size of the domain in bytes
	uint Lt_B = Lt * sizeof(float);
	uint Lx_B = Lx * sizeof(float);
	uint Ly_B = Ly * sizeof(float);
	uint Lz_B = Lz * sizeof(float);
	uint L_B = L * sizeof(float);

	// Specify the size of the domain in physical units
	float Dt;
	float Dx = 1.0;
	float Dy = 1.0;
	float Dz = 1.0;

	// Calculate the spatial step size
	float dx = Dx / (float) Lx;
	float dy = Dy / (float) Ly;
	float dz = Dz / (float) Lz;

	// calculate the temporal step size
	float gamma = 0.5;	// factor below maximum step size
	float dt = gamma * (dx * dx) / (2.0 * kappa);	// CFL condition
	Dt = dt * Lt;
	//	printf("%e\n", dt);

	// allocate pinned host memory
	float *T_h;						// dependent variables
	float *t_h , *x_h, *y_h, *z_h;	// independent variables
	CHECK(cudaMallocHost((float **) &T_h, L_B));
	CHECK(cudaMallocHost((float **) &t_h, Lt_B));
	CHECK(cudaMallocHost((float **) &x_h, Lx_B));
	CHECK(cudaMallocHost((float **) &y_h, Ly_B));
	CHECK(cudaMallocHost((float **) &z_h, Lz_B));

	// allocate device memory
	float *T_d;						// dependent variables
	float *t_d , *x_d, *y_d, *z_d;	// independent variables
	CHECK(cudaMalloc((float **) &T_d, L_B));
	CHECK(cudaMalloc((float **) &t_d, Lt_B));
	CHECK(cudaMalloc((float **) &x_d, Lx_B));
	CHECK(cudaMalloc((float **) &y_d, Ly_B));
	CHECK(cudaMalloc((float **) &z_d, Lz_B));

	// allocate cpu test memory
	float *T_cpu = new float[L];

	// initialize host memory
	int n = 0;
	for(int i = 0; i < Lx; i++){		// Initial condition for dependent variable
		float x = i * dx;
		for(int j = 0; j < Ly; j++){
			float y = j * dy;
			for(int k = 0; k < Lz; k++){
				float z = k * dz;

				// Initialize temperature as rectangle function
				if(x > 0.4f and x < 0.6f){
					T_h[n * (Lz * Ly * Lx) + k * (Ly * Lx) + j * (Lx) + i] = 10.0f;
					T_cpu[n * (Lz * Ly * Lx) + k * (Ly * Lx) + j * (Lx) + i] = 10.0f;
				} else {
					T_h[n * (Lz * Ly * Lx) + k * (Ly * Lx) + j * (Lx) + i] = 1.0f;
					T_cpu[n * (Lz * Ly * Lx) + k * (Ly * Lx) + j * (Lx) + i] = 1.0f;
				}

			}
		}
	}

	// Initialize grid
	//	memset(t_h, 0, Lt * sizeof(float));	// time grid is kept uninitialized because of CFL conditions
	for(int n = 0; n < Lt; n++) t_h[n] = n * dt;	// initialize rectangular grid in x
	for(int i = 0; i < Lx; i++) x_h[i] = i * dx;	// initialize rectangular grid in x
	for(int j = 0; j < Ly; j++) y_h[j] = j * dy;	// initialize rectangular grid in y
	for(int k = 0; k < Lz; k++) z_h[k] = k * dz;	// initialize rectangular grid in z

	//	for(int n = 0; n < Lt; n++) printf("%e\n", t_h[n]);
	//	for(int i = 0; i < Lx; i++) printf("%e\n", x_h[i]);

	// transfer data from the host to the device
	CHECK(cudaMemcpy(T_d, T_h, L_B, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(t_d, t_h, Lt_B, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(x_d, x_h, Lx_B, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(y_d, y_h, Ly_B, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(z_d, z_h, Lz_B, cudaMemcpyHostToDevice));

	struct timeval t1, t2;
	gettimeofday(&t1, 0);
	heat_1d_cpu(T_cpu, dt, x_h, Lt, Lx, m_f, m_b);
	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("cpu:  %f ms\n", time);

	dim3 threads(sx, sy, sz);
	dim3 blocks(Nx, Ny, Nz);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// main time-marching loop
	for(n = 0; n < Lt-1; n++){

		heat_1d<<<blocks, threads>>>(T_d, n, dt, x_d, Lx, sx, m_b);

		cudaDeviceSynchronize();

	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("gpu t =  %f ms, R = %f\n", milliseconds, time / milliseconds);

	///////////////////////////////////////


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	heat_1d_cuda<<<1,1>>>(blocks, threads, T_d, dt, x_h, Lt, Lx, sx, m_b);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("dyanmic gpu t =  %f ms, R = %f\n", milliseconds, time / milliseconds);

	// transfer data from the host to the device
	CHECK(cudaMemcpy(T_h, T_d, L_B, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(t_h, t_d, Lt_B, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(x_h, x_d, Lx_B, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(y_h, y_d, Ly_B, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(z_h, z_d, Lz_B, cudaMemcpyDeviceToHost));






	// open files
	FILE * meta_f = fopen("output/meta.dat", "wb");
	FILE * T_f = fopen("output/T.dat", "wb");
	FILE * t_f = fopen("output/t.dat", "wb");
	FILE * x_f = fopen("output/x.dat", "wb");
	FILE * y_f = fopen("output/y.dat", "wb");
	FILE * z_f = fopen("output/z.dat", "wb");
	FILE * T_f_cpu = fopen("output/T.cpu.dat", "wb");

	// save state variables
	fwrite(&Lt, sizeof(uint), 1, meta_f);
	fwrite(&Lx, sizeof(uint), 1, meta_f);
	fwrite(&Ly, sizeof(uint), 1, meta_f);
	fwrite(&Lz, sizeof(uint), 1, meta_f);

	// write data
	fwrite(T_h, sizeof(float), L, T_f);
	fwrite(t_h, sizeof(float), Lt, t_f);
	fwrite(x_h, sizeof(float), Lx, x_f);
	fwrite(y_h, sizeof(float), Ly, y_f);
	fwrite(z_h, sizeof(float), Lz, z_f);
	fwrite(T_cpu, sizeof(float), L, T_f_cpu);

	// close files
	fclose(meta_f);
	fclose(T_f);
	fclose(t_f);
	fclose(x_f);
	fclose(y_f);
	fclose(z_f);
	fclose(T_f_cpu);

	printf("here!");

	return 0;
}





// Compute dA/ds for for forward finite difference, first order
__device__ float D_F_1D(float A1, float A0, float s1, float s0){

	return (A1 - A0) / (s1 - s0);

}


__device__ float load_T(float * T, int n, int i, int j, int k, uint Lx, uint Ly, uint Lz){

	return T[n * (Lz * Ly * Lx) + k * (Ly * Lx) + j * (Lx) + i];

}

__device__ float load_t(float * t, int n){

	return t[n];

}

__device__ float load_z(float * z, int k){

	return z[k];

}

__device__ float load_y(float * y, int j){

	return y[j];

}

__device__ float load_x(float * x, int i){

	return x[i];

}

void print_device_properties(cudaDeviceProp prop){
	printf("%s: %s\n", "name", prop.name);                  /**< ASCII string identifying device */
	printf("%s: %lu\n", "totalGlobalMem", prop.totalGlobalMem);             /**< Global memory available on device in bytes */
	printf("%s: %lu\n", "sharedMemPerBlock", prop.sharedMemPerBlock);          /**< Shared memory available per block in bytes */
	printf("%s: %d\n", "regsPerBlock", prop.regsPerBlock);               /**< 32-bit registers available per block */
	printf("%s: %d\n", "warpSize", prop.warpSize);                   /**< Warp size in threads */
	printf("%s: %lu\n", "memPitch", prop.memPitch);                   /**< Maximum pitch in bytes allowed by memory copies */
	printf("%s: %d\n", "maxThreadsPerBlock", prop.maxThreadsPerBlock);         /**< Maximum number of threads per block */
	printf("%s: %d, %d, %d\n", "maxThreadsDim", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);           /**< Maximum size of each dimension of a block */
	printf("%s: %d, %d, %d\n", "maxGridSize", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);             /**< Maximum size of each dimension of a grid */
	printf("%s: %d\n", "clockRate", prop.clockRate);                  /**< Clock frequency in kilohertz */
	printf("%s: %lu\n", "totalConstMem", prop.totalConstMem);              /**< Constant memory available on device in bytes */
	printf("%s: %d\n", "major", prop.major);                      /**< Major compute capability */
	printf("%s: %d\n", "minor", prop.minor);                      /**< Minor compute capability */
	printf("%s: %lu\n", "textureAlignment", prop.textureAlignment);           /**< Alignment requirement for textures */
	printf("%s: %lu\n", "texturePitchAlignment", prop.texturePitchAlignment);      /**< Pitch alignment requirement for texture references bound to pitched memory */
	printf("%s: %d\n", "deviceOverlap", prop.deviceOverlap);              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
	printf("%s: %d\n", "multiProcessorCount", prop.multiProcessorCount);        /**< Number of multiprocessors on device */
	printf("%s: %d\n", "kernelExecTimeoutEnabled", prop.kernelExecTimeoutEnabled);   /**< Specified whether there is a run time limit on kernels */
	printf("%s: %d\n", "integrated", prop.integrated);                 /**< Device is integrated as opposed to discrete */
	printf("%s: %d\n", "canMapHostMemory", prop.canMapHostMemory);           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
	printf("%s: %d\n", "computeMode", prop.computeMode);                /**< Compute mode (See ::cudaComputeMode) */
	printf("%s: %d\n", "maxTexture1D", prop.maxTexture1D);               /**< Maximum 1D texture size */
	printf("%s: %d\n", "maxTexture1DMipmap", prop.maxTexture1DMipmap);         /**< Maximum 1D mipmapped texture size */
	printf("%s: %d\n", "maxTexture1DLinear", prop.maxTexture1DLinear);         /**< Maximum size for 1D textures bound to linear memory */
	printf("%s: %d, %d\n", "maxTexture2D", prop.maxTexture2D[0], prop.maxTexture2D[1]);            /**< Maximum 2D texture dimensions */
	printf("%s: %d, %d\n", "maxTexture2DMipmap", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);      /**< Maximum 2D mipmapped texture dimensions */
	printf("%s: %d, %d, %d\n", "maxTexture2DLinear", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]);      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
	printf("%s: %d, %d\n", "maxTexture2DGather", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
	printf("%s: %d, %d, %d\n", "maxTexture3D", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);            /**< Maximum 3D texture dimensions */
	printf("%s: %d, %d, %d\n", "maxTexture3DAlt", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]);         /**< Maximum alternate 3D texture dimensions */
	printf("%s: %d\n", "maxTextureCubemap", prop.maxTextureCubemap);          /**< Maximum Cubemap texture dimensions */
	printf("%s: %d, %d\n", "maxTexture1DLayered", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);     /**< Maximum 1D layered texture dimensions */
	printf("%s: %d, %d, %d\n", "maxTexture2DLayered", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);     /**< Maximum 2D layered texture dimensions */
	printf("%s: %d, %d\n", "maxTextureCubemapLayered", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);/**< Maximum Cubemap layered texture dimensions */
	printf("%s: %d\n", "maxSurface1D", prop.maxSurface1D);               /**< Maximum 1D surface size */
	printf("%s: %d, %d\n", "maxSurface2D", prop.maxSurface2D[0], prop.maxSurface2D[1]);            /**< Maximum 2D surface dimensions */
	printf("%s: %d, %d, %d\n", "maxSurface3D", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);            /**< Maximum 3D surface dimensions */
	printf("%s: %d, %d\n", "maxSurface1DLayered", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);     /**< Maximum 1D layered surface dimensions */
	printf("%s: %d, %d, %d\n", "maxSurface2DLayered", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]);     /**< Maximum 2D layered surface dimensions */
	printf("%s: %d\n", "maxSurfaceCubemap", prop.maxSurfaceCubemap);          /**< Maximum Cubemap surface dimensions */
	printf("%s: %d, %d\n", "maxSurfaceCubemapLayered", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);/**< Maximum Cubemap layered surface dimensions */
	printf("%s: %lu\n", "surfaceAlignment", prop.surfaceAlignment);           /**< Alignment requirements for surfaces */
	printf("%s: %d\n", "concurrentKernels", prop.concurrentKernels);          /**< Device can possibly execute multiple kernels concurrently */
	printf("%s: %d\n", "ECCEnabled", prop.ECCEnabled);                 /**< Device has ECC support enabled */
	printf("%s: %d\n", "pciBusID", prop.pciBusID);                   /**< PCI bus ID of the device */
	printf("%s: %d\n", "pciDeviceID", prop.pciDeviceID);                /**< PCI device ID of the device */
	printf("%s: %d\n", "pciDomainID", prop.pciDomainID);                /**< PCI domain ID of the device */
	printf("%s: %d\n", "tccDriver", prop.tccDriver);                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
	printf("%s: %d\n", "asyncEngineCount", prop.asyncEngineCount);           /**< Number of asynchronous engines */
	printf("%s: %d\n", "unifiedAddressing", prop.unifiedAddressing);          /**< Device shares a unified address space with the host */
	printf("%s: %d\n", "memoryClockRate", prop.memoryClockRate);            /**< Peak memory clock frequency in kilohertz */
	printf("%s: %d\n", "memoryBusWidth", prop.memoryBusWidth);             /**< Global memory bus width in bits */
	printf("%s: %d\n", "l2CacheSize", prop.l2CacheSize);                /**< Size of L2 cache in bytes */
	printf("%s: %d\n", "maxThreadsPerMultiProcessor", prop.maxThreadsPerMultiProcessor);/**< Maximum resident threads per multiprocessor */
	printf("%s: %d\n", "streamPrioritiesSupported", prop.streamPrioritiesSupported);  /**< Device supports stream priorities */
	printf("%s: %d\n", "globalL1CacheSupported", prop.globalL1CacheSupported);     /**< Device supports caching globals in L1 */
	printf("%s: %d\n", "localL1CacheSupported", prop.localL1CacheSupported);      /**< Device supports caching locals in L1 */
	printf("%s: %lu\n", "sharedMemPerMultiprocessor", prop.sharedMemPerMultiprocessor); /**< Shared memory available per multiprocessor in bytes */
	printf("%s: %d\n", "regsPerMultiprocessor", prop.regsPerMultiprocessor);      /**< 32-bit registers available per multiprocessor */
	printf("%s: %d\n", "managedMemory", prop.managedMemory);              /**< Device supports allocating managed memory on this system */
	printf("%s: %d\n", "isMultiGpuBoard", prop.isMultiGpuBoard);            /**< Device is on a multi-GPU board */
	printf("%s: %d\n", "multiGpuBoardGroupID", prop.multiGpuBoardGroupID);       /**< Unique identifier for a group of devices on the same multi-GPU board */
	printf("%s: %d\n", "hostNativeAtomicSupported", prop.hostNativeAtomicSupported);  /**< Link between the device and the host supports native atomic operations */
	printf("%s: %d\n", "singleToDoublePrecisionPerfRatio", prop.singleToDoublePrecisionPerfRatio); /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
	printf("%s: %d\n", "pageableMemoryAccess", prop.pageableMemoryAccess);       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
	printf("%s: %d\n", "concurrentManagedAccess", prop.concurrentManagedAccess);    /**< Device can coherently access managed memory concurrently with the CPU */
}
