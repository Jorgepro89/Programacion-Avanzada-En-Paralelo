
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLOCK_SIZE 64

template <typename T> 

__global__ void gpu_conv2_shared(T* in, int mat_size, T* kernel, int kernel_size, T* out) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * (BLOCK_SIZE - kernel_size + 1) + tx;
	int row = blockIdx.y * (BLOCK_SIZE - kernel_size + 1) + ty;
	int row_i = row - kernel_size + 1;
	int col_i = col - kernel_size + 1;

	__shared__ T in_title[BLOCK_SIZE][BLOCK_SIZE];

	if(row_i < mat_size && row_i >= 0 && col_i < mat_size && col_i)


}