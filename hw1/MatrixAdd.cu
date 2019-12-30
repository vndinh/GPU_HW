#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"
#include <cuda.h>

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

__global__ void MatrixAdd(float* A, float* B, float* S, int Width, int Height) {
	int tid, tx, ty;
	tx = threadIdx.x + blockIdx.x * blockDim.x;
	ty = threadIdx.y + blockIdx.y * blockDim.y;
	tid = Width * ty + tx;
	S[tid] = A[tid] + B[tid];
}

int main() {
	int nx = 16000;
	int ny = 16000;
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
	
	// Host memory allocation
	A = (float*)malloc(BufferSize);
	B = (float*)malloc(BufferSize);
	Sum = (float*)malloc(BufferSize);
	S_C = (float*)malloc(BufferSize);
	
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
	
	// Device memory allocation
	cudaMalloc((void**)&dev_A, BufferSize);
	cudaMalloc((void**)&dev_B, BufferSize);
	cudaMalloc((void**)&dev_S, BufferSize);
	
	// Host to device
	cudaMemcpy(dev_A, A, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, BufferSize, cudaMemcpyHostToDevice);
	
	// Kernel function
	MatrixAdd<<<grid, block>>>(dev_A, dev_B, dev_S, nx, ny);
	
	// Device to host
	cudaMemcpy(Sum, dev_S, BufferSize, cudaMemcpyDeviceToHost);
	
	// Print and check result
	MatrixAddC(A, B, S_C, MatrixHeight, MatrixWidth);
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
