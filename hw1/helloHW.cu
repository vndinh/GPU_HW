#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

__global__ void funct(void) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	if ((ix == 55) && (iy == 55)) {
		printf("Hello from GPU thread whose (ix,iy)=(%d,%d)!\n",ix,iy);
	}
}

int main(void) {
	int dimx = 4;
	int dimy = 4;
	int nx = 100;
	int ny = 100;
	dim3 block(dimx, dimy);
	dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
	funct<<<grid, block>>>();
	printf("Hello, World from CPU!\n");
	cudaDeviceReset();
	return 0;
}
