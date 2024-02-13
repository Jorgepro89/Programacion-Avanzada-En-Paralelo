
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_hello_cuda()
{
    //Aqui le quitamos la complejidad de tener un for para 
    int i = threadIdx.x;
    printf("[DEVICE] ThreadIdx: %d\n", i);
}

int main()
{
    //          aqui es donde se declara el grid
    print_hello_cuda << <2, 8 >> > ();
    return 0; 
}


