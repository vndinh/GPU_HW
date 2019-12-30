#include<cuda.h>
#include<stdio.h>
#include<cuda_runtime.h>
#include <cuda_profiler_api.h>

#define NSTREAM 8

void matricMul(int *A, int *B, int *C, int size) {
	for (int col = 0; col < size; col++) {
		for (int row = 0; row < size; row++) {
			int outidx = col * size + row;
			for (int idx = 0; idx < size; idx++) {
				C[outidx] += A[col*size + idx] * B[idx*size + row];
			}
		}
	}
}

void matrixMulCheck(int *C_cpu, int *C_gpu, int size) {
	bool ResultFlag = true;

	// Print the result
	for (int i = 0; i < size; i++) {
		if (C_cpu[i] != C_gpu[i]) {
			ResultFlag = false;
			printf("Error: C_cpu[%d] = %d; C_gpu[%d] = %d;\n", i, C_cpu[i], i, C_gpu[i]);
			break;
		}
	}
	if (ResultFlag == true) printf("Matrix Multiplication OK!\n");
	else printf("Matrix Multiplication Error!\n");
}

__global__ void matrixMulDepth(int *A, int *B, int *C, int size) {
	int tid, tx, ty;
	tx = threadIdx.x + blockDim.x * blockIdx.x;
	ty = threadIdx.y + blockDim.y * blockIdx.y;
	tid = size * ty + tx;

	int Aval = 0;
	int Bval = 0;
	int Cval = 0;

	for (int i = 0; i < size; i++) {
		Aval = A[ty * size + i];
		Bval = B[i * size + tx];
		Cval += Aval * Bval;
	}

	C[tid] = Cval;
}

int main() {
	int nx = 1600;
	int ny = 1600;
	int dimx = 32;
	int dimy = 16;

	dim3 block(dimx, dimy);	// Block dimension 32x16
	dim3 grid((nx+block.x-1)/block.x, (ny/NSTREAM+block.y-1)/block.y);

	char *iname = "CUDA_DEVICE_MAX_CONNECTIONS";
	setenv(iname, "4", 1);

	int MatrixSize = nx * ny;
	int BufferSize = MatrixSize * sizeof(int);
	int iElem = MatrixSize / NSTREAM;
	int ibytes = iElem * sizeof(int);

	int *h_A, *h_B, *h_C;
	int *C_cpu;

	// Create streams
	cudaStream_t *stream = (cudaStream_t*)malloc(NSTREAM*sizeof(cudaStream_t));
	for (int i = 0; i < NSTREAM; i++) {
		cudaStreamCreate(&stream[i]);
	}

	// Host memory allocation
	cudaHostAlloc((void**)&h_A, BufferSize, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_B, BufferSize, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_C, BufferSize, cudaHostAllocDefault);
	cudaHostAlloc((void**)&C_cpu, BufferSize, cudaHostAllocDefault);

	// Data input
	for (int i = 0; i < MatrixSize; i++) {
		h_A[i] = i % 100;
		h_B[i] = i % 100;
		h_C[i] = 0;
		C_cpu[i] = 0;
	}

	int *d_A, *d_B, *d_C;

	// Device memory allocation
	cudaMalloc((void**)&d_A, BufferSize);
	cudaMalloc((void**)&d_B, BufferSize);
	cudaMalloc((void**)&d_C, BufferSize);

	cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0);

	for (int i = 0; i < NSTREAM; i++) {
		int ioffset = i * iElem;
		cudaMemcpyAsync(&d_A[ioffset], &h_A[ioffset], ibytes, cudaMemcpyHostToDevice, stream[i]);
		matrixMulDepth<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], d_B, &d_C[ioffset], nx);
		cudaMemcpyAsync(&h_C[ioffset], &d_C[ioffset], ibytes, cudaMemcpyDeviceToHost, stream[i]);
		cudaStreamSynchronize(stream[i]);
	}

	cudaProfilerStop();

	// Check result
	matricMul(h_A, h_B, C_cpu, nx);
	matrixMulCheck(C_cpu, h_C, nx);

	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(C_cpu);
	
	return 0;
}