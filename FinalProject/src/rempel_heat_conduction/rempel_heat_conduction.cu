#include "rempel_heat_conduction.h"

#include "constants.h"


// define variables for measuring performance
//float cpu_time = 0.0;
//float gpu_time = 0.0;

int main(void)
{



	// allocate cpu memory
	float * T_cpu = new float[L];
	float * T_gpu = new float[L];
	float * x = new float[Lx];

	// initialize the grid
	initial_grid(x);

	// apply initial conditions
	initial_conditions(T_cpu, x);
	initial_conditions(T_gpu, x);



	// run CPU test
	float cpu_time = 0;
	cpu_time = heat_1d_cpu_solve(T_cpu, x, false);
	printf("cpu:  %f ms\n", cpu_time);

	// run GPU test
	float gpu_time = heat_1d_gpu_solve(T_gpu, x, false);
	printf("gpu t =  %f ms, R = %f\n", gpu_time, cpu_time / gpu_time);

	// calculate rms error
	float rms = 0.0;
	for(uint l = 0; l < L; l++) {
		rms += (T_cpu[l] - T_gpu[l]) * (T_cpu[l] - T_gpu[l]);
	}
	rms = std::sqrt(rms / (float) L);
	printf("CPU-GPU RMS error = %e \n", rms);

	// print something so we know it didn't crash somewhere
	printf("All tests completed!\n");

	return 0;
}

float heat_1d_gpu_solve(float * T, float * x, bool fickian){

	// Set up device
	int dev = 0;
	CHECK(cudaSetDevice(dev));

	// Print device and precision
	//	cudaDeviceProp prop;
	//	CHECK(cudaGetDeviceProperties(&prop, 0));
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
	CHECK(cudaMalloc((float **) &T_d, wt * Lx * sizeof(float)));
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
	float gpu_time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// main time-marching loop
	for(uint ti = 0; ti < Wt; ti++){	// downsampled resolution

		for(uint tj = 0; tj < wt; tj++){	// original resolution

			// start memory transfer
			if(tj == 0 and ti > 0){
				cudaStreamSynchronize(m_stream); // check if memory transfer is completed
				cudaMemcpyAsync(T_h + (ti * Lx), T_d, Lx * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
			}

			// perform timestep
			heat_1d_gpu_parabolic_step<<<blocks, threads, shared, k_stream>>>(T_d, x_d, tj);
			cudaStreamSynchronize(k_stream);


		}

	}


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);


	// copy to original argument pointers
	CHECK(cudaMemcpy(T, T_h, L * sizeof(float), cudaMemcpyHostToHost));

	save_results("gpu/", T, x);

	return gpu_time;

}

__global__ void heat_1d_gpu_parabolic_step(float * T, float * x, uint n){

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
		float Tn = T0 + dt_p * (T0 * T0 * sqrt(T0) * DDx_T0);

		// update global memory
		T[((n + 1) % wt) * Lx + i] = Tn;

	} else if(i == 0){
		T[((n + 1) % wt) * Lx + i] = T_left;
	} else {
		T[((n + 1) % wt) * Lx + i] = T_right;
	}


	return;

}

float heat_1d_cpu_solve(float * T, float * x, bool fickian){

	float cpu_time;
	struct timeval t1, t2;
	gettimeofday(&t1, 0);

	float * T_d = new float[wt * Lx];
	memcpy(T_d, T, Lx * sizeof(float));

	// main time-marching loop
	for(uint ti = 0; ti < Wt; ti++){	// downsampled resolution

		for(uint tj = 0; tj < wt; tj++){	// original resolution

			if(tj == 0 and ti > 0) {
				memcpy(T + (ti * Lx), T_d, Lx * sizeof(float));
			}
			heat_1d_cpu_hyperbolic_step(T, T_d, x, tj);

		}

	}

	gettimeofday(&t2, 0);
	cpu_time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;


	save_results("cpu/", T, x);

	return cpu_time;

}

void heat_1d_cpu_hyperbolic_step(float * T, float * T_d, float * x, uint n){

	// perform timestep
	for(uint i = 0; i < Lx; i++){

		if(i > 0 and i < Lx - 1){

			// Load stencil
			float T9 = T_d[n * Lx + (i - 1)];
			float T0 = T_d[n * Lx + (i + 0)];
			float T1 = T_d[n * Lx + (i + 1)];
			float Tz = T_d[((n - 1) % wt) * Lx + i];

			// Load position grid
			float x9 = x[i - 1];
			float x0 = x[i + 0];
			float x1 = x[i + 1];

			float c2 = c_h * c_h;

			// compute hyperbolic timescale
//			float tau = (T0 * T0 * sqrt(T0)) / c2;
//			float tau = (T0 * T0) / c2;
			float tau = 1/c2;

			float dx2 = ((x1 - x0) * (x0 - x9));
//			float dx2 = dx * dx;

			float dt = dt_h;
			float dt2 = dt * dt;

			// compute time-update
//			float Ta = (Tz * dx2 * (dt - 2 * tau) + 2 * (c2 * (T9 - 2 * T0 + T1) * dt2 + 2 * T0 * dx2) * tau) / (dx2 * (dt + 2 * tau));

			float Ta = (T0 * dt * dx2 + c2 * (T1 - 2 * T0 + T9) * dt2 * tau + (2 * T0 - Tz) * dx2 * tau) / (dx2 * (dt + tau));

			// update global memory
			T_d[((n + 1) % wt) * Lx + i] = Ta;

		} else if(i == 0){
			T_d[((n + 1) % wt) * Lx + i] = T_left;
		} else {
			T_d[((n + 1) % wt) * Lx + i] = T_right;
		}

		if(i > 1012){
			printf("%04d %04d %f\n", n,i, T_d[((n + 1) % wt) * Lx + i]);
		}


	}

}

void heat_1d_cpu_parabolic_step(float * T, float * T_d, float * x, uint n){

	// perform timestep
	for(uint i = 0; i < Lx; i++){

		if(i > 0 and i < Lx - 1){

			// Load stencil
			float T9 = T_d[n * Lx + (i - 1)];
			float T0 = T_d[n * Lx + (i + 0)];
			float T1 = T_d[n * Lx + (i + 1)];

			// Load position grid
			float x9 = x[i - 1];
			float x0 = x[i + 0];
			float x1 = x[i + 1];;

			// compute second derivative
			float DDx_T0 = (T9 - 2 * T0 + T1) / ((x1 - x0) * (x0 - x9));

			// compute time-update
			float Tn = T0 + dt_p * (T0 * T0 * sqrt(T0) * DDx_T0);

			// update global memory
			T_d[((n + 1) % wt) * Lx + i] = Tn;

		} else if(i == 0){
			T_d[((n + 1) % wt) * Lx + i] = T_left;
		} else {
			T_d[((n + 1) % wt) * Lx + i] = T_right;
		}



	}

}

void initial_conditions(float * T, float * x){

	// initialize host memory
	for(int n = wt - 1; n < wt + 1; n++){
		printf("%d\n",n);
		for(int i = 0; i < Lx; i++){		// Initial condition for dependent variable

			// Initialize temperature as rectangle function
//			if(x[] > 0.4f and x < 0.6f){
//				T[(n % wt) * Lx + i] = 1.0f;
//			} else {
//				T[(n % wt) * Lx + i] = 1.0f;
//			}
//
			T[(n % wt) * Lx + i] = 0.1 + 0.9 * pow(x[i],5);

//			T[(n % wt) * Lx + i] = pow(pow(0.1,3.5) + (1 - pow(0.1,3.5)) * x[i], 2 / 7);
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
