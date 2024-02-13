
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__)}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: $s $s $d\n\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void queryDevice() {
	int d_Count = 0;

	cudaGetDeviceCount(&d_Count);

	if (d_Count == 0) {
		cout << ("No CUDA device found:\n\r");
	}

	cudaDeviceProp(prop);

	for (int devNo = 0; devNo < d_Count; devNo++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, devNo);
		printf("Device Number; %d\n", devNo);
		printf("	Device Name; %s\n", prop.name);
		printf("	No. of Multiprocessors:				%d\n", prop.multiProcessorCount);
		printf("	Compute Capability:   				%d,%d\n", prop.major, prop.minor);
		printf("	Memory Clock Rate (KHz):			%d\n", prop.memoryClockRate);
		printf("	Memory Bus Rate (bits):				%d\n", prop.memoryBusWidth);
		printf("	Peak Memory Bandwith (GB/s):		%0.2f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		printf("	Total amount of Global Memory:		%dKB\n", prop.totalGlobalMem / 1024);
		printf("	Total amount of Const Memory:		%dKB\n", prop.totalConstMem / 1024);
		printf("	Total of Shared Memory per block:	%dKB\n", prop.sharedMemPerBlock / 1024);
		printf("	Total of Shared Memory per mp:		%dKB\n", prop.sharedMemPerMultiprocessor / 1024);
		printf("	Warp size:							%d\n", prop.warpSize);
		printf("	Max. threads per block:				%d\n", prop.maxThreadsPerBlock);
		printf("	Max. threads per MP:				%d\n", prop.maxThreadsPerMultiProcessor);
		printf("	Max. number if warps per MP:		%d\n", prop.maxThreadsPerMultiProcessor / 32);
		printf("	Max. Grid Size:						(%d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("	Max. block dimension:				(%d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	}
}

int main() {

	int A_rows = 480;
	int A_cols = 640;
	int A_size_bytes = A_rows * A_cols * sizeof(int);
	int* h_a = (int*)malloc(A_size_bytes);
	int* d_a, * d_b, * d_c, * d_out;

	GPUErrorAssertion(cudaMalloc((int**)&d_a, A_size_bytes));

	queryDevice();

	return EXIT_SUCCESS;
}




