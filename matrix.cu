#include<stdio.h>

#define BLOCK_SIZE 4

	
__global__ void matrixMul(int* A, int* B, int wA, int wB, int* C) {
	int bx = blockIdx.x; 
	int by = blockIdx.y;
	int tx = threadIdx.x; 
	int ty = threadIdx.y;
	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd = aBegin + wA - 1; 
	int aStep = BLOCK_SIZE; 
	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * wB;

	int Csub = 0;
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {

		__shared__ int As[BLOCK_SIZE][BLOCK_SIZE];

		__shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

//гребаные :банки: по 16 последовательных тредов, поэтому [ty][tx]		
		As[ty][tx] = A[a + wA * ty + tx]; 
		Bs[ty][tx] = B[b + wB * ty + tx];
		__syncthreads(); 

		for (int k = 0; k < BLOCK_SIZE; k++)
			Csub += As[ty][k] * Bs[k][tx];

		__syncthreads();
	}

	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] += Csub;
}


int main() {
	//int size;
	int hA, wA, hB, wB;
	int i, j;
	int* A, *B, *C;

	scanf("%d%d%d%d", &hA, &wA, &hB, &wB);

	if(wA != hB) {
		printf("E");
		return 0;
	}
	A = (int*)malloc(hA * wA * sizeof(int));
	B = (int*)malloc(hB * wB * sizeof(int));
	C = (int*)malloc(hA * wB * sizeof(int));

	for(i = 0; i < hA; i++)
		for(j = 0; j < wA; j++)
			//scanf("%d", &A[i][j]);
			scanf("%d", &A[i * wA + j]);

	printf("#############\n");
	for(i = 0; i < hB; i++)
		for(j = 0; j < wB; j++)
			scanf("%d", &B[i * hB + j]);
			
			
	printf("\n");
	int* Ad;
	cudaMalloc((void**)&Ad, hA * wA * sizeof(int));
	cudaMemcpy(Ad, A, hA * wA * sizeof(int), cudaMemcpyHostToDevice);
	
	int* Bd; 
	cudaMalloc((void**)&Bd, hB * wB * sizeof(int));
	cudaMemcpy(Bd, B, hB * wB * sizeof(int), cudaMemcpyHostToDevice);

	int* Cd;
	//size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, hA * wB * sizeof(int));

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(wB / BLOCK_SIZE, hA / BLOCK_SIZE);

	matrixMul<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);

	cudaMemcpy(C, Cd, hA * wB * sizeof(int), cudaMemcpyDeviceToHost);
	
	for(i = 0; i < hA; i++) {
		for(j = 0; j < wB; j++)
			printf("%d ", C[i * hA + j]);
		printf("\n");
	}

	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd); 
	free(A);
	free(B);
	free(C);
	return 0;	
} 
