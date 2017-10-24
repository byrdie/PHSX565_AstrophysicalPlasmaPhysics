/*
 * heat_parabolic_cuda.h
 *
 *  Created on: Oct 22, 2017
 *      Author: byrdie
 */

#ifndef HEAT_PARABOLIC_CUDA_H_
#define HEAT_PARABOLIC_CUDA_H_

#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda.h>
#include "common.h"



void print_device_properties(cudaDeviceProp prop);

__global__ void solve(float * T, float * t, float * x, float * y, float * z, uint Lt, uint Lx, uint Ly, uint Lz);
//__global__ void heat_1D(float * T, uint n, float dt, float * x, uint Lx, uint m_f, uint m_b);
__device__ float D_F_1D(float A0, float A1, float s0, float s1);
__device__ float load_T(float * T, int n, int i, int j, int k, uint Lx, uint Ly, uint Lz);
__device__ float load_t(float * t, int n);
__device__ float load_z(float * z, int k);
__device__ float load_y(float * y, int j);
__device__ float load_x(float * x, int i);

#endif /* HEAT_PARABOLIC_CUDA_H_ */
