#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <cuda_runtime.h>

#define BS 8
#define ind_l(i, j, k) (((i + l->nx) % l->nx) + ((j + l->ny) % l->ny) * (l->nx) + ((k + l->nz) % l->nz) * (l->nx * l->ny))
#define ind(i, j, k) (((i + nx) % nx) + ((j + ny) % ny) * nx + ((k + nz) % nz) * nx * ny)
#define IND(tx, ty, tz) ((tx + nx) % nx + ((ty + ny) % ny) * BS + ((tz + nz) % nz) * BS * BS)

typedef struct {
	int nx, ny, nz;
	bool *u0;
	int steps;
	int save_steps;
} life_t;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
__global__ void life_step_kernel(bool *u0, bool *u1, int nx, int ny, int nz);
void life_file_output(FILE *f, int step, int nx, int ny, int nz, bool *u0);

int main(int argc, char **argv)
{

	if (argc != 2) {
		printf("Usage: %s input file.\n", argv[0]);
		return 0;
	}

	life_t l;
	life_init(argv[1], &l);	

	bool *u0d, *u1d, *tmp;
	cudaMalloc((void **) &u0d, sizeof(bool) * l.nx*l.ny*l.nz);
	cudaMalloc((void **) &u1d, sizeof(bool) * l.nx*l.ny*l.nz);
	cudaMalloc((void **) &tmp, sizeof(bool) * l.nx*l.ny*l.nz);
	cudaMemcpy(u0d, l.u0, sizeof(bool) * l.nx*l.ny*l.nz, cudaMemcpyHostToDevice);	

	int i, x, y, z;
	//char buf[100];
	FILE *f = fopen("/home/cuda07/cuda/0_Simple/template/out.cfg", "a");
	int size_x = 0, size_y = 0, size_z = 0;
	

	for (i = 0; i < l.steps; i++) {
		life_step_kernel<<<dim3(l.nx * l.nz / BS / BS, l.ny / BS), dim3(BS, BS, BS)>>>(u0d, u1d, l.nx, l.ny, l.nz);
		tmp = u0d;
		u0d = u1d;
		u1d = tmp;
		if (i % l.save_steps == 0) {
			life_file_output(f, i, l.nx, l.ny, l.nz, l.u0);
			cudaMemcpy(l.u0, u0d, sizeof(bool) * l.nx*l.ny*l.nz, cudaMemcpyDeviceToHost);			
		}
	}

	cudaFree(u0d);
	cudaFree(u1d);
	cudaFree(tmp);
	life_free(&l);
	fclose(f);
	return 0;
}


void life_init(const char *path, life_t *l)
{
	FILE *fd = fopen(path, "r");
	assert(fd);
	assert(fscanf(fd, "%d\n", &l->steps));
	assert(fscanf(fd, "%d\n", &l->save_steps));
	printf("Steps %d, save every %d step.\n", l->steps, l->save_steps);
	assert(fscanf(fd, "%d %d %d\n", &l->nx, &l->ny, &l->nz));
	printf("Field size: %dx%dx%d\n", l->nx, l->ny, l->nz);

	l->u0 = (bool*)calloc(l->nx * l->ny * l->nz, sizeof(bool));
	
	int i, j, k, cnt;
	cnt = 0;
	while ((fscanf(fd, "%d %d %d\n", &i, &j, &k)) != EOF) {
		l->u0[ind_l(i, j, k)] = 1;
		cnt++;
	}
	printf("Loaded %d life cells.\n", cnt);
	fclose(fd);
	
	
}
	

void life_free(life_t *l)
{
	free(l->u0);
	l->nx = l->ny = l->nz = 0;
	cudaFree(u0d);
	cudaFree(u1d);
	cudaFree(tmp);
}

void life_file_output(FILE *f, int step, int nx, int ny, int nz, bool *u0)
{
	int i1, i2, i3;
	assert(f);
	fprintf(f, "Step num = %d\n", step);
	for (i3 = 0; i3 < nz; i3++) {	
		for (i2 = 0; i2 < ny; i2++) {
			for (i1 = 0; i1 < nx; i1++) {
				if (u0[ind(i1, i2, i3)] != 0) {
					fprintf(f, "%d %d %d\n", i1, i2, i3);
				}
			}
		}
	}
	fprintf(f, "---------\n");
}


__global__ void life_step_kernel(bool *u0, bool *u1, int nx, int ny, int nz) {
	
	int BX = blockIdx.x;
	int BY = blockIdx.y;
	int BZ = blockIdx.z;
	int TX = threadIdx.x;
	int TY = threadIdx.y;
	int TZ = threadIdx.z;

	int idx = BS*BX + TX;
	int idy = BS*BY + TY;
	int idz = BS*BZ + TZ;

	int n = 0;
	
	__shared__ bool u0s[BS*BS*BS];
	u0s[IND(TX, TY, TZ)] = u0[ind(idx, idy, idz)]; 

	bool alive = u0s[IND(TX, TY, TZ)]; 
	bool alive_now = 0;	


	__syncthreads();
	
	n += l->u0[ind(idx+1, idy, idz)];

	n += l->u0[ind(idx+1, idy+1, idz)];

	n += l->u0[ind(idx,   idy+1, idz)];

	n += l->u0[ind(idx-1, idy, idz)];

	n += l->u0[ind(idx-1, idy-1, idz)];

	n += l->u0[ind(idx,   idy-1, idz)];

	n += l->u0[ind(idx-1, idy+1, idz)];

	
	n += l->u0[ind(idx+1, idy-1, idz)];
	
	n += l->u0[ind(idx+1, idy, idz-1)];

	n += l->u0[ind(idx+1, idy+1, idz-1)];

	n += l->u0[ind(idx,   idy+1, idz-1)];

	n += l->u0[ind(idx-1, idy, idz-1)];

	n += l->u0[ind(idx-1, idy-1, idz-1)];

	n += l->u0[ind(idx,   idy-1, idz-1)];	

	n += l->u0[ind(idx-1, idy+1, idz-1)];

	n += l->u0[ind(idx+1, idy-1, idz-1)];

	n += l->u0[ind(idx,   idy, idz-1)];


	n += l->u0[ind(idx+1, idy, idz+1)];

	n += l->u0[ind(idx+1, idy+1, idz+1)];

	n += l->u0[ind(idx,   idy+1, idz+1)];

	n += l->u0[ind(idx-1, idy, idz+1)];

	n += l->u0[ind(idx-1, idy-1, idz+1)];

	n += l->u0[ind(idx,   idy-1, idz+1)];

	n += l->u0[ind(idx-1, idy+1, idz+1)];

	n += l->u0[ind(idx+1, idy-1, idz+1)];

	n += l->u0[ind(idx,   idy, idz+1)];

	if ((n == 6 || n == 7) && !alive) {
		alive_now = 1;
	}
	if ((n == 4 || n == 5 || n == 6 || n == 7) && alive) {
		alive_now = 1;
	}
	
	u1[ind(idx, idy, idz)] = alive_now;
}
