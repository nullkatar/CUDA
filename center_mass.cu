#include <stdlib.h>
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

__global__ void pi_kern(CUDA_FLOAT *res)
{
        int n = threadIdx.x + blockIdx.x * BLOCK_SIZE;
        CUDA_FLOAT x0 = n * 1.f / (BLOCK_SIZE * GRID_SIZE);
        CUDA_FLOAT y0 = sqrt(1 - x0 * x0);
        CUDA_FLOAT dx = 1.f / (1.f * BLOCK_SIZE * GRID_SIZE);
        CUDA_FLOAT s = 0;
        CUDA_FLOAT x1, y1;
//      for(int i = 0; i < THREAD_SIZE; i++){
                x1 = x0 + dx; 
                y1 = sqrt(1 - x1 * x1);
                s += (y0 + y1) * dx / 2.f;
                x0 = x1;
                y0 = y1;
//      }
        res[n] = s;
}


__global__ void pi_kern_(CUDA_FLOAT *res_)
{
        int n = threadIdx.x + blockIdx.x * BLOCK_SIZE;
        CUDA_FLOAT x0 = n * 1.f / (BLOCK_SIZE * GRID_SIZE);
        CUDA_FLOAT y0 = x0 * sqrt(1 - x0 * x0);
        CUDA_FLOAT dx = 1.f / (1.f * BLOCK_SIZE * GRID_SIZE);
        CUDA_FLOAT s = 0;
        CUDA_FLOAT x1, y1;

//      for(int i = 0; i < THREAD_SIZE; i++){
        x1 = x0 + dx;
        y1 = x1 * sqrt(1 - x1 * x1);
        s += (y0 + y1) * dx / 2.f;
        x0 = x1;
        y0 = y1;
//      }
        res_[n] = s;
}


int main() {
        float *res_h,  *res_d_;
        float *res_d;
        res_h = (float *)malloc(sizeof(float)*THREAD_SIZE);
        cudaMalloc((void **) &res_d, sizeof(float)*THREAD_SIZE);
        cudaMalloc((void **) &res_d_, sizeof(float)*THREAD_SIZE);
        cudaMemcpy(res_d, res_h, sizeof(float)*THREAD_SIZE, cudaMemcpyHostToDevice);
        dim3 grid(GRID_SIZE);
        dim3 block(BLOCK_SIZE);
        pi_kern<<< grid,  block >>>(res_d);
        cudaMemcpy(res_h, res_d, sizeof(float)*THREAD_SIZE, cudaMemcpyDeviceToHost);

        float sum = 0.0;
        for(int i = 0; i < THREAD_SIZE; i++){
                sum += res_h[i];
        }

        cudaMemcpy(res_d_, res_h, sizeof(float)*THREAD_SIZE, cudaMemcpyHostToDevice);
        pi_kern_<<< grid,  block >>>(res_d_);
        cudaMemcpy(res_h, res_d_, sizeof(float)*THREAD_SIZE, cudaMemcpyDeviceToHost);

        float sum_ = 0.0;
        for(int j = 0; j < THREAD_SIZE; j++){
                sum_ += res_h[j];
        }

        printf("%0.8f\n", sum*4);
        printf("%0.8f\n", sum_ / sum);

        free(res_h);
        cudaFree(res_d);
        cudaFree(res_d_);
        return 0;

}

