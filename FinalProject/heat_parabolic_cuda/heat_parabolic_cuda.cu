#include "heat_parabolic_cuda.h"

#include "constants.h"


// define variables for measuring performance
float cpu_time = 0.0;
float gpu_time = 0.0;

int main(void)
{



	// allocate cpu memory
	float * T_cpu = new float[L];
	float * T_gpu = new float[L];
	float * x = new float[Lx];

	// apply initial conditions
	initial_conditions(T_cpu);
	initial_conditions(T_gpu);

	// initialize the grid
	initial_grid(x);

	// run CPU test
	heat_1d_cpu_solve(T_cpu, x);

	// run GPU test
	heat_1d_gpu_solve(T_gpu, x);

	// calculate rms error
	float rms = 0.0;
	for(uint l = 0; l < L; l++) {
		rms += (T_cpu[l] - T_gpu[l]) * (T_cpu[l] - T_gpu[l]);
	}
	rms = std::sqrt(rms / (float) L);
	printf("CPU-GPU RMS error = %e \n", rms);
	//	if(divergence == false){
	//		printf("CPU/GPU match\n");
	//	} else {
	//		printf("CPU/GPU DIVERGENCE!!!!!!!!!!\n");
	//	}

	// print something so we know it didn't crash somewhere
	printf("All tests completed!\n");

	return 0;
}

void heat_1d_gpu_solve(float * T, float * x){

	// Set up device
	int dev = 0;
	CHECK(cudaSetDevice(dev));

	// Print device and precision
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, 0));
	//		print_device_properties(prop);

	// configure the device to have the largest possible L1 cache
	CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// allocate pinned host memory
	float *T_h;						// dependent variables
	float *x_h;	// independent variables
	CHECK(cudaMallocHost((float **) &T_h, L * sizeof(float)));
	CHECK(cudaMallocHost((float **) &x_h, Lx * sizeof(float)));

	// allocate device memory
	float *T_d;						// dependent variables
	float *x_d;	// independent variables
	CHECK(cudaMalloc((float **) &T_d, L * sizeof(float)));
	CHECK(cudaMalloc((float **) &x_d, Lx * sizeof(float)));

	// transfer initial condition from argument to pinned host memory
	CHECK(cudaMemcpy(T_h, T, Lx * sizeof(float), cudaMemcpyHostToHost));
	CHECK(cudaMemcpy(x_h, x, Lx * sizeof(float), cudaMemcpyHostToHost));

	// transfer data from the host to the device
	CHECK(cudaMemcpy(T_d, T_h, Lx * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(x_d, x_h, Lx * sizeof(float), cudaMemcpyHostToDevice));

	// set the number of threads and blocks
	const uint threads = lx;
	const uint blocks = Nx;

	// set the amount of shared memory
	const uint shared = 0;

	// initialize streams
	cudaStream_t k_stream, m_stream;
	cudaStreamCreate(&k_stream);	// initialize computation stream
	cudaStreamCreate(&m_stream);	// initialize memory stream

	// initialize timing events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// main time-marching loop
	for(uint ti = 1; ti < Wt; ti++){	// downsampled resolution

		// check if memory transfer is completed
		cudaStreamSynchronize(m_stream);

		for(uint tj = 0; tj < wt; tj++){	// original resolution

			// perform timestep
			heat_1d_device_step<<<blocks, threads, shared, k_stream>>>(T_d, x_d, tj);
			cudaStreamSynchronize(k_stream);

			// start memory transfer
			if(tj == 0){
				cudaMemcpyAsync(T_h + (ti * Lx), T_d, Lx * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
			}


		}



	}


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	printf("gpu t =  %f ms, R = %f\n", gpu_time, cpu_time / gpu_time);


	// copy to original argument pointers
	CHECK(cudaMemcpy(T, T_h, Wt * Lx * sizeof(float), cudaMemcpyHostToHost));

	save_results("gpu/", T, x);

}

__global__ void heat_1d_device_step(float * T, float * x, uint n){

	// Find index from threadId
	uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i > 0 and i < (Lx - 1)){

		// Load stencil
		float T9 = T[n * Lx + (i - 1)];
		float T0 = T[n * Lx + (i + 0)];
		float T1 = T[n * Lx + (i + 1)];

		// Load position grid
		float x9 = x[i - 1];
		float x0 = x[i + 0];
		float x1 = x[i + 1];

		// compute Laplacian
		float DDx_T0 = (T9 - 2 * T0 + T1) / ((x1 - x0) * (x0 - x9));

		// compute time-update
		float Tn = T0 + dt * (kappa * DDx_T0);

		// update global memory
		T[((n + 1) % wt) * Lx + i] = Tn;

	} else {
		T[((n + 1) % wt) * Lx + i] = 0;
	}


	return;

}

__global__ void heat_1d_shfl_step(float * T, float * x, uint n){

	// Find index from threadId
	uint i = blockIdx.x * blockDim.x + threadIdx.x;

	uint laneID = threadIdx.x % 32;

	float T0 = T[n * Lx + i];
	float x0 = x[i];

	float T1, T9;
	float x1, x9;

	T1 = __shfl_down(T0, 1);
	T9 = __shfl_up(T0, 1);
	x1 = __shfl_down(x0, 1);
	x9 = __shfl_up(x0, 1);

	if(i > 0 and i < (Lx - 1)){

		if(laneID == 0){

			T9 = T[n * Lx + (i - 1)];
			x9 = x[i - 1];

		} else if (laneID == 31) {

			T1 = T[n * Lx + (i + 1)];
			x1 = x[i + 1];

		}

		// compute Laplacian
		float DDx_T0 = (T9 - 2 * T0 + T1) / ((x1 - x0) * (x0 - x9));

		// compute time-update
		float Tn = T0 + dt * (kappa * DDx_T0);

		// update global memory
		T[(n + 1) * Lx + i] = Tn;

	} else {
		T[(n + 1) * Lx + i] = 0;
	}

}

__global__ void heat_1d_shared_step(float * T, float * x, uint n){

	// allocate shared memory
	__shared__ float T_s[lx];
	__shared__ float x_s[lx];

	// Find index from threadId
	uint i = blockIdx.x * sx + threadIdx.x;

	// load data from device memory into shared memory
	T_s[threadIdx.x] = T[n * Lx + i];
	x_s[threadIdx.x] = x[i];

	__syncthreads();

	// domain boundary
	if(i > 0 and i < (Lx - 1)){

		// shared memory stencil boundary
		if (threadIdx.x > 0 and threadIdx.x < (lx - 1)) {

			// Load stencil
			float T9 = T_s[threadIdx.x - 1];
			float T0 = T_s[threadIdx.x + 0];
			float T1 = T_s[threadIdx.x + 1];

			// Load position grid
			float x9 = x_s[threadIdx.x - 1];
			float x0 = x_s[threadIdx.x + 0];
			float x1 = x_s[threadIdx.x + 1];

			// compute Laplacian
			float DDx_T0 = (T9 - 2 * T0 + T1) / ((x1 - x0) * (x0 - x9));

			// compute time-update
			float Tn = T0 + dt * (kappa * DDx_T0);

			// update global memory
			T[(n + 1) * Lx + i] = Tn;

		}

	} else {	// boundary condition
		T[(n + 1) * Lx + i] = 0;
	}

	return;

}





void heat_1d_cpu_solve(float * T, float * x){

	struct timeval t1, t2;
	gettimeofday(&t1, 0);

	float * T_d = new float[wt * Lx];
	memcpy(T_d, T, Lx * sizeof(float));

	// main time-marching loop
	for(uint ti = 1; ti < Wt; ti++){	// downsampled resolution

		for(uint tj = 0; tj < wt; tj++){	// original resolution

			// perform timestep
			for(uint i = 0; i < Lx; i++){

				if(i != 0 and i != Lx - 1){

					// Load stencil
					float T9 = T_d[tj * Lx + (i - 1)];
					float T0 = T_d[tj * Lx + (i + 0)];
					float T1 = T_d[tj * Lx + (i + 1)];

					// Load position grid
					float x9 = x[i - 1];
					float x0 = x[i + 0];
					float x1 = x[i + 1];;

					// compute second derivative
					float DDx_T0 = (T9 - 2 * T0 + T1) / ((x1 - x0) * (x0 - x9));

					// compute time-update
					float Tn = T0 + dt * (kappa * DDx_T0);

					// update global memory
					T_d[((tj + 1) % wt) * Lx + i] = Tn;

					if(tj == 0) {
						T[ti * Lx + i] = Tn;
					}

				} else {	// boundary condition

					T_d[((tj + 1) % wt) * Lx + i] = 0;

					if(tj == 0) {
						T[ti * Lx + i] = 0;
					}

				}

			}

//			// downsample operation
//			if(tj == 0){
//				memcpy(T + (ti * Lx), T_d, Lx * sizeof(float));
//			}

		}

	}

	gettimeofday(&t2, 0);
	cpu_time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
	printf("cpu:  %f ms\n", cpu_time);

	save_results("cpu/", T, x);

	return;

}

void initial_conditions(float * T){

	// initialize host memory
	int n = 0;
	for(int i = 0; i < Lx; i++){		// Initial condition for dependent variable
		float x = i * dx;

		// Initialize temperature as rectangle function
		if(x > 0.4f and x < 0.6f){
			T[n * Lx + i] = 10.0f;
		} else {
			T[n * Lx + i] = 1.0f;
		}


	}

}

void initial_grid(float * x){

	for(int i = 0; i < Lx; i++) x[i] = i * dx;	// initialize rectangular grid in x

}

void save_results(std::string path, float * T,  float * x){

	// open files
	FILE * meta_f = fopen(("output/" + path + "meta.dat").c_str(), "wb");
	FILE * T_f = fopen(("output/" + path + "T.dat").c_str(), "wb");
	FILE * x_f = fopen(("output/" + path + "x.dat").c_str(), "wb");

	// save state variables
	fwrite(&Lt, sizeof(uint), 1, meta_f);
	fwrite(&Lx, sizeof(uint), 1, meta_f);

	// write data
	fwrite(T, sizeof(float), L, T_f);
	fwrite(x, sizeof(float), Lx, x_f);

	// close files
	fclose(meta_f);
	fclose(T_f);
	fclose(x_f);


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
