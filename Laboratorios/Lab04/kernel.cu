#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define WIDTH 1024
#define HEIGHT 1024

void convolution2D(int* mat, int* res, int width, int height) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int sum = 0;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int curRow = row + i;
                    int curCol = col + j;
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        sum += mat[curRow * width + curCol];
                    }
                }
            }
            res[row * width + col] = sum;
        }
    }
}

int main() {
    const int dataSize = WIDTH * HEIGHT * sizeof(int);

    int* M = new int[WIDTH * HEIGHT];
    int* M_res = new int[WIDTH * HEIGHT];

    // Initialize matrix M with random values
    srand(time(NULL));
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        M[i] = rand() % 9;
    }

    std::cout << "Before: \n";
    for (int i = 0; i < 20; ++i) {
        std::cout << "M[" << i << "] = " << M[i] << std::endl;
    }

    convolution2D(M, M_res, WIDTH, HEIGHT);

    std::cout << "After: \n";
    for (int i = 0; i < 20; ++i) {
        std::cout << "RES[" << i << "] = " << M_res[i] << std::endl;
    }

    delete[] M;
    delete[] M_res;

    return 0;
}