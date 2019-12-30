#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define DIM 4096

extern __shared__ int dsmem[];

// Recursive implementation of interleaved pair approach
int recursiveReduce(int *data, int const size) {
  if (size == 1) return data[0];
  int const stride = size / 2;
  for (int i = 0; i < stride; i++)
    data[i] += data[i + stride];

  return recursiveReduce(data, stride);
}

__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int n) {
  // Set thread ID
  unsigned int tid = threadIdx.x;
  int *idata = g_idata + blockIdx.x * blockDim.x;

  // Boundary check
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // In-place reduction in global memory
  if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
  __syncthreads();

  // Unrolling the last warp
  if (tid < 32) {
    volatile int *vsmem = idata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // Write result for this block to global memory
  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int n) {
  __shared__ int smem[DIM];

  // Set thread ID
  unsigned int tid = threadIdx.x;

  // Boundary check
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // Convert global pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;

  // Load data to shared memory
  smem[tid] = idata[tid];
  __syncthreads();

  // In-place reduction in shared memory
  if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
  __syncthreads();

  // Unrolling the last warp
  if (tid < 32) {
    volatile int *vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // Write result to global memory for this block
  if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemDyn(int *g_idata, int *g_odata, unsigned int n) {
  extern __shared__ int smem[];

  // Set thread ID
  unsigned int tid = threadIdx.x;

  // Convert global pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;

  // Load data from global memory to shared memory
  smem[tid] = idata[tid];
  __syncthreads();

  // In-place reduction in shared memory
  if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
  __syncthreads();

  // Unrolling the last warp
  if (tid < 32) {
    volatile int *vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // Write result to global memory for this block
  if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceGmemUnroll(int *g_idata, int *g_odata, unsigned int n) {
  // Set thread ID
  unsigned int tid = threadIdx.x;

  // 4 blocks of input data processed at a time
  unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
  int *idata = g_idata + blockIdx.x * blockDim.x * 4;

  // Unrolling data block by 4
  if (idx + 3 * blockDim.x < n) {
    int a0 = g_idata[idx];
    int a1 = g_idata[idx + blockDim.x];
    int a2 = g_idata[idx + 2 * blockDim.x];
    int a3 = g_idata[idx + 3 * blockDim.x];
    g_idata[idx] = a0 + a1 + a2 + a3;
  }
  __syncthreads();

  // In-place reduction in global memory
  if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
  __syncthreads();

  // Unrolling the last warp
  if (tid < 32) {
    volatile int *vsmem = idata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // Write result to global memory for this block
   if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmemUnroll(int *g_idata, int *g_odata, unsigned int n) {
  // Static shared memory
  __shared__ int smem[DIM];

  // Set thread ID
  unsigned int tid = threadIdx.x;

  // 4 blocks of input data processed at a time
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;

  // Unrolling 4 blocks
  int tmpSum = 0;
  if (idx + 4 * blockDim.x <= n) {
    int a0 = g_idata[idx];
    int a1 = g_idata[idx + blockDim.x];
    int a2 = g_idata[idx + 2 * blockDim.x];
    int a3 = g_idata[idx + 3 * blockDim.x];
    tmpSum = a0 + a1 + a2 + a3;
  }

  smem[tid] = tmpSum;
  __syncthreads();

  // In-place reduction in shared memory
  if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
  __syncthreads();

  // Unrolling the last warp
  if (tid < 32) {
    volatile int *vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // Write result to global memory for this block
  if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemUnrollDyn(int *g_idata, int *g_odata, unsigned int n) {
  extern __shared__ int smem[];

  // Set thread ID
  unsigned int tid = threadIdx.x;

  // 4 blocks of input data processed at a time
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;

  // Unrolling 4 blocks
  int tmpSum = 0;
  if (idx + 3 * blockDim.x < n) {
    int a0 = g_idata[idx];
    int a1 = g_idata[idx + blockDim.x];
    int a2 = g_idata[idx + 2 * blockDim.x];
    int a3 = g_idata[idx + 3 * blockDim.x];
    tmpSum = a0 + a1 + a2 + a3;
  }

  smem[tid] = tmpSum;
  __syncthreads();

  // In-place reduction in global memory
  if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
  __syncthreads();

  // Unrolling the last warp
  if (tid < 32) {
    volatile int *vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // Write result to global memory for this block
  if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceNeighboredGmem(int *g_idata, int *g_odata, unsigned int n) {
  // Set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Convert global pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;

  // Boundary check
  if (idx >= n) return;

  // In-place reduction in global memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2*stride)) == 0) idata[tid] += idata[tid + stride];
    __syncthreads();
  }

  // Write result to global memory for this block
  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredSmem(int *g_idata, int *g_odata, unsigned int n) {
  // Static shared memory
  __shared__ int smem[DIM];

  // Set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Convert global pointer to local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;

  // Boundary check
  if (idx >= n) return;

  smem[tid] = idata[tid];
  __syncthreads();

  // In-place reduction in shared memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2*stride)) == 0) smem[tid] = smem[tid + stride];
    __syncthreads();
  }

  // Write result to global memory for this block
  if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

int main(int argc, char **argv) {
  // Set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("%s starting reduction at ", argv[0]);
  printf("device %d: %s ", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  bool bResult = false;

  // Initialization
  int power = 12;

  // Execution configuration
  int blockSize = DIM;
  if (argc >= 2) blockSize = atoi(argv[1]);
  if (argc >= 3) power = atoi(argv[2]);

  // Total number of elements to reduce
  int size = 1 << power; 
  printf(" with array size %d ", size);

  dim3 block (blockSize, 1);
  dim3 grid ((size + block.x - 1) / block.x, 1);
  printf("grid %d, block %d \n", grid.x, block.x);

  // Allocate host memory
  size_t ibytes = size * sizeof(int);
  size_t obytes = grid.x * sizeof(int);

  int *h_idata = (int*)malloc(ibytes);
  int *h_odata = (int*)malloc(obytes);
  int *tmp = (int*)malloc(ibytes);

  // Initialize the array
  for (int i = 0; i < size; i++) h_idata[i] = (int)(rand() & 0xFF);

  memcpy(tmp, h_idata, ibytes);

  int gpu_sum = 0;

  // Allocate device memory
  int *d_idata = NULL;
  int *d_odata = NULL;
  CHECK(cudaMalloc((void**)&d_idata, ibytes));
  CHECK(cudaMalloc((void**)&d_odata, obytes));

  // CPU reduction
  int cpu_sum = recursiveReduce(tmp, size);
  printf("CPU reduce: %d \n", cpu_sum);

  // Reduce Global Memory
  CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
  reduceGmem<<<grid, block>>>(d_idata, d_odata, size);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
  printf("reduceGmem <<<grid %d, block %d>>>: %d\t\t", grid.x, block.x, gpu_sum);
  bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Failed!\n");
  else printf("OK!\n");

  // Reduce Static Shared Memory
  CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
  reduceSmem<<<grid, block>>>(d_idata, d_odata, size);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
  printf("reduceSmem <<<grid %d, block %d>>>: %d\t\t", grid.x, block.x, gpu_sum);
  bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Failed!\n");
  else printf("OK!\n");

  // Reduce Dynamic Shared Memory
  CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
  reduceSmemDyn<<<grid, block, blockSize*sizeof(int)>>>(d_idata, d_odata, size);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
  printf("reduceSmemDyn <<<grid %d, block %d>>>: %d\t\t", grid.x, block.x, gpu_sum);
  bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Failed!\n");
  else printf("OK!\n");

  // Reduce Global Memory with Unrolling
  CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
  reduceGmemUnroll<<<grid.x/4, block>>>(d_idata, d_odata, size);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_odata, d_odata, obytes/4, cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x/4; i++) gpu_sum += h_odata[i];
  printf("reduceGmemUnroll <<<grid %d, block %d>>>: %d\t\t", grid.x/4, block.x, gpu_sum);
  bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Failed!\n");
  else printf("OK!\n");

  // Reduce Static Shared Memory with Unrolling
  CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
  reduceSmemUnroll<<<grid.x/4, block>>>(d_idata, d_odata, size);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_odata, d_odata, obytes/4, cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x/4; i++) gpu_sum += h_odata[i];
  printf("reduceSmemUnroll <<<grid %d, block %d>>>: %d\t\t", grid.x/4, block.x, gpu_sum);
  bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Failed!\n");
  else printf("OK!\n");

  // Reduce Dynamic Shared Memory with Unrolling
  CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
  reduceSmemUnrollDyn<<<grid.x/4, block, DIM*sizeof(int)>>>(d_idata, d_odata, size);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_odata, d_odata, obytes/4, cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x/4; i++) gpu_sum += h_odata[i];
  printf("reduceSmemUnrollDyn <<<grid %d, block %d>>>: %d\t\t", grid.x/4, block.x, gpu_sum);
  bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Failed!\n");
  else printf("OK!\n");

  // Reduce Neighbored Global Memory
  CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
  reduceNeighboredGmem<<<grid, block>>>(d_idata, d_odata, size);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
  printf("reduceNeighboredGmem <<<grid %d, block %d>>>: %d\t\t", grid.x, block.x, gpu_sum);
  bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Failed!\n");
  else printf("OK!\n");

  // Reduce Neighbored Shared Memory
  CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
  reduceNeighboredSmem<<<grid, block>>>(d_idata, d_odata, size);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
  printf("reduceNeighboredSmem <<<grid %d, block %d>>>: %d\t\t", grid.x, block.x, gpu_sum);
  bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Failed!\n");
  else printf("OK!\n");

  // Free host memory
  free(h_idata);
  free(h_odata);

  // Free device memory
  CHECK(cudaFree(d_idata));
  CHECK(cudaFree(d_odata));

  // Reset device
  CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}
