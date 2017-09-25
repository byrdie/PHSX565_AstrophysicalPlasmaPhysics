#include <stdio.h>
#include <assert.h>
#include <cuda.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.


int main(void)
{
  // Print device and precision
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("\nDevice Name: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);


  int Nx = 256;
  int Ny = 256;
  int Nt = 64;
  int N = Nx * Ny * Nt;

  // host arrays
  float * f_H = new float[N];

  // device arrays
  float * f_D;
  cudaMalloc((void**) &f_D, N * sizeof(float));


  return 0;
}
