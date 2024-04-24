#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iostream>
#include <cstdlib>
#include <ctime>

void transpose(int* input, int* output, int width, int height) {
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            int index_in = y * width + x;
            int index_out = x * height + y;
            output[index_out] = input[index_in];
        }
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int dataSize = width * height * sizeof(int);

    int* matrix = new int[width * height];
    int* transposed = new int[width * height];

    // Initialize matrix with random values
    srand(time(NULL));
    for (int i = 0; i < width * height; ++i) {
        matrix[i] = rand() % 9;
    }

    std::cout << "Original M: \n";
    for (int i = 0; i < 20; ++i) {
        std::cout << "M[" << i << "] = " << matrix[i] << std::endl;
    }

    transpose(matrix, transposed, width, height);

    std::cout << "Transposed M: \n";
    for (int i = 0; i < 20; ++i) {
        std::cout << "M Transposed[" << i << "] = " << transposed[i] << std::endl;
    }

    delete[] matrix;
    delete[] transposed;

    return 0;
}