#include "common.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define BDIMX 16

void printData(int *in, int size) {
	for (int i = 0; i < size; i++) printf("%2d ", in[i]);
	printf("\n");
}

__global__ void test_shfl_wrap(int *dout, int *din, int offset) {
	int value = din[threadIdx.x];
	value = __shfl(value, threadIdx.x + offset, BDIMX);
	dout[threadIdx.x] = value;
}

__global__ void test_shfl_wrap_plus(int *dout, int *din, int offset) {
	int value = din[threadIdx.x];
	value += __shfl(value, threadIdx.x + offset, BDIMX);
	dout[threadIdx.x] = value;
}

int main(int argc, char **argv) {
	int dev = 0;
	bool iPrintout = 1;

	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("> %s Starting.", argv[0]);
	printf("at Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	int nElem = BDIMX;
	int h_din[BDIMX], h_dout[BDIMX];

	for (int i = 0; i < nElem; i++) {
		h_din[i] = i;
	}

	if (iPrintout) {
		printf("Initial data\t\t: ");
		printData(h_din, nElem);
	}

	size_t nBytes = nElem * sizeof(int);
	int *d_din, *d_dout;
	CHECK(cudaMalloc((int**)&d_din, nBytes));
	CHECK(cudaMalloc((int**)&d_dout, nBytes));

	CHECK(cudaMemcpy(d_din, h_din, nBytes, cudaMemcpyHostToDevice));

	int block = BDIMX;

	test_shfl_wrap<<<1, block>>>(d_dout, d_din, -2);
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(h_dout, d_dout, nBytes, cudaMemcpyDeviceToHost));
	if (iPrintout) {
		printf("test_shfl_wrap\t\t: ");
		printData(h_dout, nElem);
	}

	test_shfl_wrap_plus<<<1, block>>>(d_dout, d_din, 2);
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(h_dout, d_dout, nBytes, cudaMemcpyDeviceToHost));
	if (iPrintout) {
		printf("test_shfl_wrap_plus\t: ");
		printData(h_dout, nElem);
	}

	CHECK(cudaFree(d_din));
	CHECK(cudaFree(d_dout));
	CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;
}
