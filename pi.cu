#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <locale.h>
#include <cuda.h>

#define BLOCK_SIZE 250
#define GRID_SIZE 4
#define THREAD_SIZE 1000
#define CUDA_FLOAT float

__global__ void pi_kern(double *res)
{
        int n = threadIdx.x + blockIdx.x * BLOCK_SIZE;
        CUDA_FLOAT x0 = n * 1.f / (BLOCK_SIZE * GRID_SIZE);
        CUDA_FLOAT y0 = sqrtf(1 - x0 * x0);
        CUDA_FLOAT dx = 1.f / (1.f * BLOCK_SIZE * GRID_SIZE * THREAD_SIZE);
        CUDA_FLOAT s = 0;
        CUDA_FLOAT x1, y1;
        x1 = x0 + dx;
        y1 = sqrtf(1 - x1 * x1);
        s += (y0 + y1) * dx / 2.f;
        x0 = x1;
        y0 = y1;

        res[n] = s;
}

int main() {
        double *res_h, *res_d;
        res_h = (double *)malloc(sizeof(double)*THREAD_SIZE);
        cudaMalloc((void **) &res_d, sizeof(double)*THREAD_SIZE);
        cudaMemcpy(res_d, res_h, sizeof(double)*THREAD_SIZE, cudaMemcpyHostToDevice);
        pi_kern<<< GRID_SIZE, BLOCK_SIZE >>>(res_d);
        cudaMemcpy(res_h, res_d, sizeof(double)*THREAD_SIZE, cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for(int i = 0; i < THREAD_SIZE; i++){
                sum += res_h[i];
        }

        printf("%0.8f\n", sum / THREAD_SIZE);

        printf("%d", 2);
        free(res_h);
        cudaFree(res_d);
        return 0;
}

