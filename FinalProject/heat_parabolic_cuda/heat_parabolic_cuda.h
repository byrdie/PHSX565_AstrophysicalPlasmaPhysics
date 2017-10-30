/*
 * heat_parabolic_cuda.h
 *
 *  Created on: Oct 22, 2017
 *      Author: byrdie
 */

#ifndef HEAT_PARABOLIC_CUDA_H_
#define HEAT_PARABOLIC_CUDA_H_

#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <assert.h>
#include <cmath>
#include <cuda.h>
#include "common.h"





void print_device_properties(cudaDeviceProp prop);

void initial_conditions(float * T);
void initial_grid(float * t, float * x);

void heat_1d_cpu_solve(float * T, float * t, float * x);

void heat_1d_gpu_solve(float * T, float * t, float * x);

__global__ void heat_1d_device_step(float * T, float * x, uint n);
__global__ void heat_1d_shared_step(float * T, float * x, uint n);
__global__ void heat_1d_shfl_step(float * T, float * x, uint n);

void save_results(std::string path, float * T, float * t, float * x);

#endif /* HEAT_PARABOLIC_CUDA_H_ */
