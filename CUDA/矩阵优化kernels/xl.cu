#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

//block-thread 3D-3D
__global__ void testBlockThread9(int* c, const int* a, const int* b) {
    int threadId_3D = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int blockId_3D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int i = threadId_3D + (blockDim.x * blockDim.y * blockDim.z) * blockId_3D;
    c[i] = b[i] - a[i];
}


void addWithCuda(int* c, const int* a, const int* b, unsigned int size) {
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    cudaSetDevice(0);

    cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    uint3 s1; s1.x = 5; s1.y = 2; s1.z = 2;
    uint3 s2; s2.x = size / 200; s2.y = 5; s2.z = 2;
    testBlockThread9 << <s1, s2 >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaGetLastError();
}


int main() {
    const int n = 1000;

    int* a = new int[n];
    int* b = new int[n];
    int* c = new int[n];
    int* cc = new int[n];

    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
        c[i] = b[i] - a[i];
    }

    addWithCuda(cc, a, b, n);

    FILE* fp = fopen("out.txt", "w");
    for (int i = 0; i < n; i++)
        fprintf(fp, "%d %d\n", c[i], cc[i]);
    fclose(fp);

    bool flag = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != cc[i]) {
            flag = false;
            break;
        }
    }

    if (flag == false)
        printf("no pass");
    else
    {
        int sum = 0;
        sum=5/12;
        printf("sum=%d",sum);
        printf("pass");
    }


    cudaDeviceReset();

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] cc;

    getchar();
    return 0;
}
