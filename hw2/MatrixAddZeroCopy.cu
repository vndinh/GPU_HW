#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

void MatrixAddC(float* A, float* B, float* S, int Width, int Height) {
  int col = 0;
  int row = 0;
  int DestIndex = 0;
  for (col = 0; col < Width; col++) {
    for (row = 0; row < Height; row++) {
      DestIndex = col * Width + row;
      S[DestIndex] = A[DestIndex] + B[DestIndex];
    }   
  }   
}

__global__ void MatrixAddZeroCopy(float* A, float* B, float* S, int Width, int Height) {
  int tid, tx, ty; 
  tx = threadIdx.x + blockIdx.x * blockDim.x;
  ty = threadIdx.y + blockIdx.y * blockDim.y;
  tid = Width * ty + tx; 
  S[tid] = A[tid] + B[tid];
}

int main() {
  int nx = 160;
  int ny = 160;
  int dimx = 32; 
  int dimy = 32; 
    
  dim3 block(dimx, dimy);
  dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    
  const int MatrixSize = nx * ny; 
  const int BufferSize = MatrixSize * sizeof(float);
    
  float* A;
  float* B;
  float* Sum;
  float* S_C;
    
  // Zero-Copy memory allocation
  cudaHostAlloc((void**)&A, BufferSize, cudaHostAllocMapped);
  cudaHostAlloc((void**)&B, BufferSize, cudaHostAllocMapped);
  cudaHostAlloc((void**)&Sum, BufferSize, cudaHostAllocMapped);
  cudaHostAlloc((void**)&S_C, BufferSize, cudaHostAllocMapped);
    
  // Data input
  for (int i = 0; i < MatrixSize; i++) {
    A[i] = i;
    B[i] = i;
    Sum[i] = 0;
    S_C[i] = 0;
  }

  float* dev_A;
  float* dev_B;
  float* dev_S;

  // Get device pointer
  cudaHostGetDevicePointer((void**)&dev_A, (void*)A, 0);
  cudaHostGetDevicePointer((void**)&dev_B, (void*)B, 0);
  cudaHostGetDevicePointer((void**)&dev_S, (void*)Sum, 0);

  // Kernel function
  MatrixAddZeroCopy <<<grid, block>>>(dev_A, dev_B, dev_S, nx, ny);

  // Synchronize threads
  cudaThreadSynchronize();

  cudaProfilerStop();

  // Print and check result
  MatrixAddC(A, B, S_C, nx, ny);
  bool ResultFlag = true;
  for (int i = 0; i < MatrixSize; i++) {
    if (Sum[i] != S_C[i]) {
      ResultFlag = false;
      printf("Error Matrix Add at element %d\n", i);
      break;
    }
  }
  if (ResultFlag == true) printf("Matrix Add is OK\n");
  else printf("Error Matrix Add\n");

  // Free memory
  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_S);

  free(A);
  free(B);
  free(Sum);
  free(S_C);

  return 0;
}
                                  