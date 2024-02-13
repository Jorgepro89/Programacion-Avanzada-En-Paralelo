
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void print_all_idx()
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bidz = blockIdx.z;

    int gidx = gridDim.x;
    int gidy = gridDim.y;
    int gidz = gridDim.z;

    printf("[DEVICE] threadIdx.x: %d, blockIdx.x: %d, gridDim.x: %d \n", tidx, bidx, gidx);
    printf("[DEVICE] threadIdx.y: %d, blockIdx.y: %d, gridDim.y: %d \n", tidy, bidy, gidx);
    printf("[DEVICE] threadIdx.z: %d, blockIdx.z: %d, gridDim.z: %d \n", tidz, bidz, gidz);
}

int main()
{
    dim3 blockSize(4, 4, 4);
    dim3 gridSize(2, 2, 2);

    //Punteros (Saber cuales usar en GPU y en CPU)
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

    //Almacenar el la memoria nuestras variables
    //Ayuda a reservar memoria en la memoria de video
    // Memory allocation
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);
    
    // Transfer CPU host to GPU device
    cudaMemcpy(c_device, c_host, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_device, a_host, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, data_size, cudaMemcpyHostToDevice);

    // Launch to kernel
    print_all_idx << <gridSize , blockSize >> > (); 

    // Transfer GPU device to CPU host
    cudaMemcpy(c_device, c_host, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_device, a_host, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_device, b_host, data_size, cudaMemcpyDeviceToHost);

    //Limpieza de la memoria
    cudaDeviceReset();
    cudaFree(c_device);
    cudaFree(a_device);
    cudaFree(b_device);

    return 0;
}

/*int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}*/
