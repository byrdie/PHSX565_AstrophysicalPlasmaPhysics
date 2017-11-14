#include "rempel_heat_conduction.h"

#include "constants.h"


int main(void)
{



	// allocate cpu memory
	float * T_cpu = new float[L];
	float * T_gpu = new float[L];
	float * q_cpu = new float[L];
	float * q_gpu = new float[L];
	float * x = new float[Lx];

	// initialize the grid
	initial_grid(x);

	// apply initial conditions
	initial_conditions(T_cpu, q_cpu, x);
	initial_conditions(T_gpu, q_gpu, x);



	// hyperbolic test
	float cpu_time = 0;
	cpu_time = heat_1d_cpu_solve(T_cpu, q_cpu, x, false, "cpu/");
	printf("cpu:  %f ms\n", cpu_time);

	// parabolic test
	float gpu_time = 0;
	gpu_time = heat_1d_cpu_solve(T_gpu, q_gpu, x, true, "gpu/");
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


float heat_1d_cpu_solve(float * T, float * q, float * x, bool fickian, std::string path){

	printf(" dx = %e\n dt_p = %e\n dt_h = %e\n c_h = %e\n tau = %e\n", dx, dt_p, dt_h, c_h, 1/ (c_h * c_h));

	float cpu_time;
	struct timeval t1, t2;
	gettimeofday(&t1, 0);

	float * T_d = new float[wt * Lx];
	float * q_d = new float[wt * Lx];
	memcpy(T_d, T, Lx * sizeof(float));
	memcpy(q_d, q, Lx * sizeof(float));

	// main time-marching loop
	for(uint ti = 0; ti < Wt; ti++){	// downsampled resolution

		for(uint tj = 0; tj < wt; tj++){	// original resolution

			if(tj == 0 and ti > 0) {
				memcpy(T + (ti * Lx), T_d, Lx * sizeof(float));
				memcpy(q + (ti * Lx), q_d, Lx * sizeof(float));
			}

			if(fickian){
				heat_1d_cpu_parabolic_step(T, T_d, q_d, x, tj);
			} else {
				heat_1d_cpu_hyperbolic_step(T, T_d, q_d, x, tj);
			}


			//

		}

	}

	gettimeofday(&t2, 0);
	cpu_time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;


	save_results(path, T, x);

	return cpu_time;

}

void heat_1d_cpu_hyperbolic_step(float * T, float * T_d, float * q, float * x, uint n){


	// perform timestep
	uint i;
	for(i = 0; i < (Lx - 1); i++){

		// Load stencil
		float T0 = T_d[n * Lx + (i + 0)];
		float T1 = T_d[n * Lx + (i + 1)];


		float q0 = q[n * Lx + (i + 0)];

		// Load position grid

		float x0 = x[i + 0];
		float x1 = x[i + 1];

		float c2 = c_h * c_h;

		float kappa = (T0 * T0 * sqrt(T0) + T1 * T1 * sqrt(T1)) / 2.0;

		//		 compute hyperbolic timescale

				float tau = kappa / c2;




		float dt = dt_h;


//		tau = max(tau, 4.0*dt);


//		float qa = q0 - dt * (q0 + kappa * (T1 - T0) /  dx) / tau;		// explict
		float qa = dt * (q0 * tau / dt - kappa * (T1 - T0) / dx) / (tau + dt); // implicit
		q[((n + 1) % wt) * Lx + i] = qa;

		if(i > 0) {
			float q9 = q[n * Lx + (i - 1)];
			float x9 = x[i - 1];
			float Ta = T0 - ((q0 - q9) * dt / (x0 - x9));
			T_d[((n + 1) % wt) * Lx + i] = Ta;
		}

	}

	// apply left boundary conditions
	i = 0;
	T_d[((n + 1) % wt) * Lx + i] = T_left;


	// apply right boundary conditions
	i = Lx - 1;
	T_d[((n + 1) % wt) * Lx + i] = T_right;
}

void heat_1d_cpu_parabolic_step(float * T, float * T_d, float * q, float * x, uint n){

	// perform timestep
	uint i;
	for(i = 0; i < Lx - 1; i++){

		// Load stencil
		float T0 = T_d[n * Lx + (i + 0)];
		float T1 = T_d[n * Lx + (i + 1)];


		float q0 = q[n * Lx + (i + 0)];

		// Load position grid

		float x0 = x[i + 0];
		float x1 = x[i + 1];

		float kappa = (T0 * T0 * sqrt(T0) + T1 * T1 * sqrt(T1)) / 2.0;


		float dt = dt_p;

		float qa = -kappa * (T1 - T0) / (x1 - x0);
		q[((n + 1) % wt) * Lx + i] = qa;

		if(i > 0) {
			float q9 = q[n * Lx + (i - 1)];
			float x9 = x[i - 1];
			float Ta = T0 - ((q0 - q9) * dt / (x0 - x9));
			T_d[((n + 1) % wt) * Lx + i] = Ta;
		}

	}

	// apply left boundary conditions
	i = 0;
	T_d[((n + 1) % wt) * Lx + i] = T_left;


	// apply right boundary conditions
	i = Lx - 1;
	T_d[((n + 1) % wt) * Lx + i] = T_right;
}

void initial_conditions(float * T, float * q, float * x){

	int n = 0;

	// initialize host memory
	printf("%d\n",n);
	for(int i = 0; i < Lx; i++){		// Initial condition for dependent variable

		float T0 = 0.1 + 0.9 * pow(x[i],5);
		float T1 = 0.1 + 0.9 * pow(x[i + 1],5);

		T[(n % wt) * Lx + i] = T0;

		q[n * Lx + i] = 0;



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
