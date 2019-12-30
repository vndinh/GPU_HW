#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

__device__ int d_islower(char c) {
  if (c > 96 && c < 123) return 1;
  else return 0;
}

__device__ int d_toupper(char c) {
  if (c > 64 && c < 91) return c;
  else return c - 32;
}

__global__ void toUpperString(char* c) {
  if (d_islower(c[blockIdx.x])) c[blockIdx.x] = d_toupper(c[blockIdx.x]);
}

int main() {
  const int N = 16;
  char* c;

  int BufferSize = N * sizeof(char);

  // Host memory allocation
  c = (char*)malloc(BufferSize);

  // Assign value to string
  strcpy(c, "good luck! guys");

  char* dev_C;

  // Device memory allocation
  cudaMalloc((void**)&dev_C, BufferSize);

  // Copy string from host to device
  cudaMemcpy(dev_C, c, BufferSize, cudaMemcpyHostToDevice);

  // Kernel function
  toUpperString<<<N, 1>>>(dev_C);

  // Copy upper string from device to host
  cudaMemcpy(c, dev_C, BufferSize, cudaMemcpyDeviceToHost);

  printf("%s\n", c);

  // Free memory
  cudaFree(dev_C);
  free(c);

  return 0;
}
