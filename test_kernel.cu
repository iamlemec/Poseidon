#include <stdio.h>
#include <cuda.h>

// Kernel that executes on the CUDA device
__global__ void square_array(double *a, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) a[idx] = a[idx] * a[idx];
}

void run_square_array(double* a_d, int N)
{
  int block_size = 4;
  int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);

  // Retrieve result from device and store it in host array
  square_array<<<n_blocks,block_size>>>(a_d,N);
  cudaError_t code = cudaGetLastError();
  printf("Poseidon_kernel error: %s\n", cudaGetErrorString(code));
}
