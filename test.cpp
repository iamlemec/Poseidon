#include <stdio.h>
#include <malloc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "test_kernel.h"

// main routine that executes on the host
int main(void)
{
  const int N = 10;
  size_t size = N*sizeof(double);

  double* a_h = (double*)malloc(size);
  double* a_d;
  cudaMalloc((void**)&a_d, size);

  // Initialize host array and copy it to CUDA device
  for (int i = 0; i < N; i++) {
    a_h[i] = (double)i;
  }
  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

  // Do calculation on device:
  run_square_array(a_d,N);

  // Retrieve result from device and store it in host array
  cudaMemcpy(a_h, a_d, sizeof(double)*N, cudaMemcpyDeviceToHost);

  // Print results
  for (int i = 0; i < N; i++) {
    printf("%d %f\n", i, a_h[i]);
  }

  // Cleanup
  free(a_h);
  cudaFree(a_d);
}
