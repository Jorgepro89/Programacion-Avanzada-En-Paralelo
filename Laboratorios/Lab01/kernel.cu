
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

__global__ void printGlobalIds()
{
    int id = ((blockDim.y * blockIdx.y) + threadIdx.y) * (gridDim.x * blockDim.x) + (blockDim.x * blockIdx.x) + threadIdx.x;
    printf("[DEVICE] Global Id: %d\n", id);
}

int main()
{
    dim3 blockSize(4, 2, 1);
    dim3 gridSize(2, 2, 1);

    // Punteros (Saber cuales usar en GPU y en CPU)
    // Initialization
    int* c_host; //Host o CPU
    int* a_host;
    int* b_host;

    int* c_device; //Device o GPU
    int* a_device;
    int* b_device;

    const int data_count = 10000;
    const int data_size = data_count * sizeof(int);

    c_host = (int*)malloc(data_size);
    a_host = (int*)malloc(data_size);
    b_host = (int*)malloc(data_size);

    // Almacenar el la memoria nuestras variables
    // Ayuda a reservar memoria en la memoria de video
    // Memory allocation
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);

    // Transfer CPU host to GPU device
    cudaMemcpy(c_device, c_host, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_device, a_host, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, data_size, cudaMemcpyHostToDevice);

    // Launch to kernel
    printGlobalIds << <gridSize, blockSize >> > ();

    // Transfer GPU device to CPU host
    cudaMemcpy(c_device, c_host, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_device, a_host, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_device, b_host, data_size, cudaMemcpyDeviceToHost);

    // Limpieza de la memoria
    cudaDeviceReset();
    cudaFree(c_device);
    cudaFree(a_device);
    cudaFree(b_device);

    return 0;
}