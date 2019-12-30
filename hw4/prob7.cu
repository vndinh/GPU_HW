#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BDIMX 32
#define BDIMY 16
#define IPAD 2

void printData(char *msg, int *in, const int size) {
	printf("%s: ", msg);
	for (int i = 0; i < size; i++) {
		printf("%4d", in[i]);
		fflush(stdout);
	}
	printf("\n");
}

__global__ void setRowReadColPad(int *out) {
	// Static shared memory
	__shared__ int tile[BDIMY][BDIMX+IPAD];

	// Mapping from 2-D thread index to 1-D global memory address
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	// Convert idx to transposed (irow, icol)
	unsigned int irow = idx / blockDim.y;
	unsigned int icol = idx % blockDim.y;

	// Transfer data from global memory to shared memory
	tile[threadIdx.y][threadIdx.x] = idx;

	// Wait for all threads completed
	__syncthreads();

	// Transfer data from shared memory to global memory
	out[idx] = tile[icol][irow];
}

__global__ void setColReadRowPad(int *out) {
	// Static shared memory
	__shared__ int tile[BDIMX][BDIMY+IPAD];

	// Mapping form 2-D thread index to 1-D global memory address
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	// Convert idx to transposed (irow, icol)
	unsigned int irow = idx / blockDim.y;
	unsigned int icol = idx % blockDim.y;

	// Transfer data from global memory to shared memory
	tile[threadIdx.x][threadIdx.y] = idx;

	// Wait for all threads completed
	__syncthreads();

	// Transfer data from shared memory to global memory
	out[idx] = tile[irow][icol];
}

int main(int argc, char **argv) {
	// Set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s at ", argv[0]);
	printf("device %d: %s ", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	cudaSharedMemConfig pConfig;
	CHECK(cudaDeviceGetSharedMemConfig(&pConfig));
	printf("with Bank Mode: %s ", pConfig == 1 ? "4-byte" : "8-byte");

	// Setup array size
	int nx = BDIMX;
	int ny = BDIMY;

	bool iprintf = 0;

	if (argc > 1) iprintf = atoi(argv[1]);

	size_t nBytes = nx * ny * sizeof(int);

	// Execution configuration
	dim3 block (BDIMX, BDIMY);
	dim3 grid (1, 1);
	printf("<<<grid(%d, %d), block(%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Allocate device memory
	int *d_C;
	CHECK(cudaMalloc((int**)&d_C, nBytes));
	int *gpuRef = (int *)malloc(nBytes);

	CHECK(cudaMemset(d_C, 0, nBytes));
	setRowReadColPad<<<grid, block>>>(d_C);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	if (iprintf) printData("setRowReadColPad ", gpuRef, nx*ny);

	CHECK(cudaMemset(d_C, 0, nBytes));
	setColReadRowPad<<<grid, block>>>(d_C);
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	if (iprintf) printData("setColReadRowPad ", gpuRef, nx*ny);

	CHECK(cudaFree(d_C));
	free(gpuRef);

	// Reset device
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}