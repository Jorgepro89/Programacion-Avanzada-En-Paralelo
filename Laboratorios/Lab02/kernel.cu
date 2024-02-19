#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

__global__ void printGlobalIds()
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;

    int threadId = id_x + id_y * gridDim.x * blockDim.x + id_z * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    printf("[DEVICE] ThreadIdx: (%d, %d, %d), BlockIdx: (%d, %d, %d), GridDim: (%d, %d, %d), Global Thread ID: %d\n",
        threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z, threadId);
}

int main()
{
    dim3 blockSize(2, 2, 2); // Tamaño de block de 2 hilos en cada dimensión
    dim3 gridSize(2, 2, 2);   // Tamaño de grid de 2 bloques en cada dimensión

    // Launch kernel
    printGlobalIds << <gridSize, blockSize >> > ();

    // Sincronizar para asegurar que el kernel termina antes de salir
    cudaDeviceSynchronize();

    // Limpieza de la memoria
    cudaDeviceReset();

    return 0;
}
