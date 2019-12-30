#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_WIDTH 32

void matricMul(int *A, int *B, int *C, int size) {
	for (int col = 0; col < size; col++) {
		for (int row = 0; row < size; row++){
			int outidx = col * size + row;
			for (int idx = 0; idx < size; idx++)
				C[outidx] += A[col*size+idx] * B[idx*size+row];
		}
	}
}

void matrixMulCheck(int *C_test, int *C_cuda, int size) {
	bool ResultFlag = true;

	// Print the result
	for (int i = 0; i < size; i++) {
		if (C_test[i] != C_cuda[i]) {
			ResultFlag = false;
			printf("Error: C_test[%d] = %d; C_cuda[%d] = %d;\n", i, C_test[i], i, C_cuda[i]);
			break;
		}
	}
	if (ResultFlag == true) printf("Matrix Multiplication OK!\n");
	else printf("Matrix Multiplication Error!\n");
}

__global__ void matrixMulGSmem(int *A, int *B, int *C, int size) {

	// Static shared memory
	__shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * TILE_WIDTH + ty;
	int col = blockIdx.x * TILE_WIDTH + tx;

	int Cval = 0;

	for (int i = 0; i < (size/TILE_WIDTH); i++) {
		if ((row < size) && (i*TILE_WIDTH+tx < size)) ds_A[ty][tx] = A[row*size+i*TILE_WIDTH+tx];
		else ds_A[ty][tx] = 0;

		if ((col < size) && (i*TILE_WIDTH+ty < size)) ds_B[ty][tx] = B[col+size*(i*TILE_WIDTH+ty)];
		else ds_B[ty][tx] = 0;

		__syncthreads();

		for (int j = 0; j < TILE_WIDTH; j++) Cval += ds_A[ty][j] * ds_B[j][tx];

		__syncthreads();
	}

	if (row < size && col < size) C[row * size + col] = Cval;
}

int main() {
	int nx = 1600;
	int ny = 1600;
	int dimx = 32;
	int dimy = 16;

	dim3 block(dimx, dimy);	// Block dimension 32x16
	dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

	int MatrixSize = nx * ny;
	int BufferSize = MatrixSize * sizeof(int);

	int *h_A;
	int *h_B;
	int *h_C;
	int *C_test;

	// Host memory allocation
	h_A = (int*)malloc(BufferSize);
	h_B = (int*)malloc(BufferSize);
	h_C = (int*)malloc(BufferSize);
	C_test = (int*)malloc(BufferSize);

	// Data input
	for (int i = 0; i < nx; i++) {
		h_A[i] = i % 100;
		h_B[i] = i % 100;
		h_C[i] = 0;
		C_test[i] = 0;
	}

	int *d_A;
	int *d_B;
	int *d_C;

	// Device memory allocation
	cudaMalloc((void**)&d_A, BufferSize);
	cudaMalloc((void**)&d_B, BufferSize);
	cudaMalloc((void**)&d_C, BufferSize);

	// Copy data from Host to Device
	cudaMemcpy(d_A, h_A, BufferSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, BufferSize, cudaMemcpyHostToDevice);

	// Matrix Multiplication
	matrixMulGSmem<<<grid, block, TILE_WIDTH*TILE_WIDTH*sizeof(int)>>>(d_A, d_B, d_C, nx);

	// Copy result from Device to Host
	cudaMemcpy(h_C, d_C, BufferSize, cudaMemcpyDeviceToHost);

	// Check result
	matricMul(h_A, h_B, C_test, nx);
	matrixMulCheck(C_test, h_C, nx);

	// Free memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_C);
	free(C_test);
	
	return 0;
}
