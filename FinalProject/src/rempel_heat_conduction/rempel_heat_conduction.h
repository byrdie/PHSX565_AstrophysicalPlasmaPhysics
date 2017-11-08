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







void initial_conditions(float * T, float * q, float * x);
void initial_grid(float * x);

float heat_1d_cpu_solve(float * T, float * q, float * x, bool fickian);

void heat_1d_cpu_parabolic_step(float * T, float * T_d, float * x, uint n);
void heat_1d_cpu_hyperbolic_step(float * T, float * T_d, float * q, float * x, uint n);

float heat_1d_gpu_solve(float * T, float * q, float * x, bool fickian);

__global__ void heat_1d_gpu_parabolic_step(float * T, float * x, uint n);
__global__ void heat_1d_gpu_hyperbolic_step(float * T, float * x, uint n);

void save_results(std::string path, float * T, float * x);

#endif /* HEAT_PARABOLIC_CUDA_H_ */
