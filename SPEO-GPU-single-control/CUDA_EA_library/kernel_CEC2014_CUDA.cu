#include "CEC2014_CUDA.h"
#include "device_launch_parameters.h"

extern __shared__ real sh_mem_CEC2014[];

static __device__ __forceinline__ int device_ParallelSum(real * vector)
{
	__syncwarp();	
	if (threadIdx.x < WARP_SIZE / 2)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 2];
	__syncwarp();	
	if (threadIdx.x < WARP_SIZE / 4)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 4];
	__syncwarp();	
	if (threadIdx.x < WARP_SIZE / 8)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 8];
	__syncwarp();	
	if (threadIdx.x < WARP_SIZE / 16)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 16];
	__syncwarp();	
	if (threadIdx.x < WARP_SIZE / 32)
		vector[threadIdx.x] += vector[threadIdx.x + WARP_SIZE / 32];

	return 0;

};
static __device__ __forceinline__ int device_ParallelMultiple(real * vector)
{
	__syncwarp();	
	if (threadIdx.x < WARP_SIZE / 2)
		vector[threadIdx.x] *= vector[threadIdx.x + WARP_SIZE / 2];
	__syncwarp();	
	if (threadIdx.x < WARP_SIZE / 4)
		vector[threadIdx.x] *= vector[threadIdx.x + WARP_SIZE / 4];
	__syncwarp();		
	if (threadIdx.x < WARP_SIZE / 8)
		vector[threadIdx.x] *= vector[threadIdx.x + WARP_SIZE / 8];
	__syncwarp();		
	if (threadIdx.x < WARP_SIZE / 16)
		vector[threadIdx.x] *= vector[threadIdx.x + WARP_SIZE / 16];
	__syncwarp();		
	if (threadIdx.x < WARP_SIZE / 32)
		vector[threadIdx.x] *= vector[threadIdx.x + WARP_SIZE / 32];
	return 0;
};

static __device__ __host__ __forceinline__ int device_NextPow2Dim(int dim)
{
	int next_pow2_dim = 1;
	while (next_pow2_dim < dim)
		next_pow2_dim <<= 1;
	if (next_pow2_dim < WARP_SIZE)
		next_pow2_dim = WARP_SIZE;
	return next_pow2_dim;
};
int Configuration(dim3 *grid, dim3 *block, int * shared_mem_size, int pop_size)
{
	int temp_value = pop_size;
	if (temp_value > MAX_THREAD_PER_BLOCK / WARP_SIZE / 2)
		temp_value = MAX_THREAD_PER_BLOCK / WARP_SIZE / 2;

	block->x = WARP_SIZE;
	block->y = temp_value;
	block->z = 1;

	grid->x = 1;
	grid->y = pop_size / temp_value;
	grid->z = 1;

	*shared_mem_size = sizeof(real) * WARP_SIZE * temp_value;
	return 0;

}
__global__ void global_CalW(real *d_wi, real * d_orginal_elements, real * d_shift_data, real *d_delta, int problem_dim, int num_function)
{
	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	int loop_times = next_pow2_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;

	int index_dim = 0;
	real tmp_value = 0;
	real sum_wi = 0;

	for (int i = 0; i < num_function; i++)
	{
		sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;
		for (int j = 0; j < loop_times; j++)
		{
			index_dim = threadIdx.x + j * blockDim.x;
			if (index_dim < problem_dim)
			{
				tmp_value = d_orginal_elements[index_dim + ind_individual * next_pow2_dim];
				tmp_value = tmp_value - d_shift_data[index_dim + i * next_pow2_dim];
				tmp_value = tmp_value * tmp_value;
				sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value;
			}
		}

		real tmp_sum_value = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
		tmp_sum_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];
		if (threadIdx.x == 0)
		{
			tmp_value = tmp_sum_value;
			if (tmp_value == 0)
			{
				d_wi[i + ind_individual * num_function] = 1e30;
			}
			else
				d_wi[i + ind_individual * num_function] = 1 / sqrt(tmp_value) * exp(-tmp_value / (2 * problem_dim * d_delta[i] * d_delta[i]));
			sum_wi += d_wi[i + ind_individual * num_function];
		}
	}
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < num_function; i++)
		{
			real tmp = d_wi[i + ind_individual * num_function] / sum_wi;
			d_wi[i + ind_individual * num_function] = tmp;
		}
	}

}
int CalW(real *d_wi, real * d_orginal_elements, real * d_shift_data, real *d_delta, int problem_dim, int pop_size, int num_function)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	global_CalW << <grid, block, shared_mem_size >> >(d_wi, d_orginal_elements, d_shift_data, d_delta, problem_dim, num_function);
	return 0;

}

__global__ void global_Shuffle(real * d_shuffled_elements, real * d_original_elements, int * d_shuffle_data, int dim)
{
	int next_pow2_dim = device_NextPow2Dim(dim);
	int loop_times = next_pow2_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		if (index_dim < dim)
			d_shuffled_elements[index_dim + ind_individual * next_pow2_dim] = d_original_elements[d_shuffle_data[index_dim] - 1 + ind_individual * next_pow2_dim];
	}
}
__global__ void global_ShiftAndRotation(real * A, real * B, real * C, real * d_shift, int num_A_rows, int num_A_columns, int num_B_rows, int num_B_columns, int num_C_rows, int num_C_columns, int shift_flag)
{
	__shared__ real ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ real ds_N[TILE_WIDTH][TILE_WIDTH];
	//printf("%lf\n", A[89*32 + 1]);
	int bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		row = by * TILE_WIDTH + ty,
		col = bx * TILE_WIDTH + tx;

	real Pvalue = 0;
	for (int m = 0; m < (num_A_columns - 1) / TILE_WIDTH + 1; ++m)
	{
		if (row < num_A_rows && m * TILE_WIDTH + tx < num_A_columns)
			ds_M[ty][tx] = A[row * num_A_columns + m * TILE_WIDTH + tx] - shift_flag * d_shift[m * TILE_WIDTH + tx];
		else
			ds_M[ty][tx] = 0;
		if (col < num_B_columns && m * TILE_WIDTH + ty < num_B_rows)
			ds_N[ty][tx] = B[(m * TILE_WIDTH + ty) * num_B_columns + col];


		else
			ds_N[ty][tx] = 0;
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += ds_M[ty][k] * ds_N[k][tx];
		__syncthreads();
	}
	if (row < num_C_rows && col < num_C_columns)
	{
		C[row * num_C_columns + col] = Pvalue;

	}
	__syncthreads();
}

__global__ void global_Shift(real * d_shifted_elements, real * d_original_elements, real * d_shift_data, int pop_size, int problem_dim, int shift_flag)
{
	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	int loop_times = next_pow2_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		d_shifted_elements[index_dim + ind_individual * next_pow2_dim] = d_original_elements[index_dim + ind_individual * next_pow2_dim] - shift_flag * d_shift_data[index_dim];
	}
}

__global__ void global_CombineFitness(real * d_fitness_value, real * d_tmp_fitness_value, int num_functions, real * d_wi, int pop_size, real wi, real bais)
{
	int ind_individual = threadIdx.x + blockIdx.x * blockDim.x;
	int index_dim = 0;
	d_fitness_value[ind_individual] = 0;

	for (int i = 0; i < num_functions; i++)
		d_fitness_value[ind_individual] = d_fitness_value[ind_individual] + d_wi[i + ind_individual * num_functions] * d_tmp_fitness_value[ind_individual + i * pop_size];
	d_fitness_value[ind_individual] = wi * d_fitness_value[ind_individual] + bais;
}

int CombineFitness(real * d_fitness_value, real * d_tmp_fitness_value, int num_functions, real * d_wi, int pop_size, real wi, real bais)
{
	dim3 grid, block;
	block.x = pop_size;
	if (block.x> MAX_THREAD_PER_BLOCK / 2)
		block.x = MAX_THREAD_PER_BLOCK / 2;
	block.y = 1;
	block.z = 1;

	grid.x = pop_size / block.x;
	grid.y = 1;
	grid.z = 1;
	global_CombineFitness << <grid, block >> >(d_fitness_value, d_tmp_fitness_value, num_functions, d_wi, pop_size, wi, bais);

	return 0;
}
int ShiftAndRotation(real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data, int problem_dim, int pop_size, int shift_flag)
{
	int next_pow2_dim = device_NextPow2Dim(problem_dim);

	dim3 grid((problem_dim - 1) / TILE_WIDTH + 1, (pop_size - 1) / TILE_WIDTH + 1, 1);
	dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
	int size_shared_mem_per_block = TILE_WIDTH * TILE_WIDTH * sizeof(real);
	int size_register_per_block = 8 * TILE_WIDTH * TILE_WIDTH * 2;
	global_ShiftAndRotation << <grid, block, size_shared_mem_per_block >> >(d_original_elements, d_rotated_data, d_rotated_elements, d_shift_data, pop_size, next_pow2_dim, next_pow2_dim, problem_dim, pop_size, next_pow2_dim, shift_flag);
	return 0;

}
int Shift(real * d_shifted_elements, real * d_original_elements, real * d_shift_data, int problem_dim, int pop_size, int shift_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	global_Shift << <grid, block >> >(d_shifted_elements, d_original_elements, d_shift_data, pop_size, problem_dim, shift_flag);
	return 0;
}
int Shuffle(real * d_shuffled_elements, real * d_original_elements, int * d_shuffle_data, int problem_dim, int pop_size)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	global_Shuffle << <grid, block >> >(d_shuffled_elements, d_original_elements, d_shuffle_data, problem_dim);
	return 0;

}

__global__ void global_f1(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);

	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;

	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;

		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;
			tmp_value = pow(1000000.0, index_dim / (calcuate_dim - 1.0)) * tmp_value * tmp_value;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value;
		}
	}

	device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	if (threadIdx.x == 0)
	{

		real tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
};
__global__ void global_f2(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);

	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;

		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;
			tmp_value = 1000000.0 * tmp_value * tmp_value;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value;
		}
	}

	real tmp_fitness_value = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];
	if (threadIdx.x == 0)
	{
		tmp_value = d_elements[ind_individual * next_pow2_problem_dim] * d_elements[ind_individual * next_pow2_problem_dim];

		tmp_fitness_value = tmp_fitness_value - 1000000.0 * tmp_value + tmp_value;
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
};
__global__ void global_f3(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);

	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;

		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;
			tmp_value = tmp_value * tmp_value;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value;
		}
	}

	real tmp_fitness_value = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];
	if (threadIdx.x == 0)
	{
		tmp_value = d_elements[ind_individual * next_pow2_problem_dim] * d_elements[ind_individual * next_pow2_problem_dim];

		tmp_fitness_value = tmp_fitness_value + 1000000.0 * tmp_value - tmp_value;
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
};
__global__ void global_f4(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value1, tmp_value2;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		if (index_dim < calcuate_dim - 1)
		{
			tmp_value1 = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value2 = d_elements[index_dim + 1 + ind_individual * next_pow2_problem_dim];
			tmp_value1 = tmp_value1 * a_rate + shift;
			tmp_value2 = tmp_value2 * a_rate + shift;

			tmp_value2 = tmp_value1 * tmp_value1 - tmp_value2;
			tmp_value2 = 100.0 * tmp_value2 * tmp_value2 + (tmp_value1 - 1.0) * (tmp_value1 - 1.0);
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value2;
		}
	}

	real tmp_fitness_value = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];

	if (threadIdx.x == 0)
	{
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
};
__global__ void global_f5(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;

		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;
			tmp_value = cos(2 * M_PI * tmp_value);
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value;
		}
	}

	real tmp_fitness_value1 = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value1 = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];

	if (threadIdx.x == 0)
	{
		tmp_fitness_value1 = tmp_fitness_value1 / (calcuate_dim + 0.0);
		tmp_fitness_value1 = exp(tmp_fitness_value1);
	}
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;
	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;

		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;
			tmp_value = tmp_value * tmp_value;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value;
		}
	}

	real tmp_fitness_value2 = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value2 = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];
	if (threadIdx.x == 0)
	{
		tmp_fitness_value2 = -0.2 * sqrt(tmp_fitness_value2 / (calcuate_dim + 0.0));
		tmp_fitness_value2 = -20.0 * exp(tmp_fitness_value2);
		d_fitness_value[ind_individual] = wi * (tmp_fitness_value2 - tmp_fitness_value1 + 20 + M_E) + bias;
	}
};
__global__ void global_f6(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;

	real sinValue = 0, cosValue = 0;
	real a = 0.5;
	real b = 3.0;
	int kmax = 20;
	real fix_value = 0;
	real pow_a = 1.0;
	real pow_b = 1.0;

	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;

		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;
			real sum_value1 = 0;
			pow_a = 1.0;
			pow_b = 1.0;
			for (int i = 0; i <= kmax; i++)
			{
				fix_value = 2 * M_PI * pow_b;
				sincos(fix_value * (tmp_value + 0.5), &sinValue, &cosValue);
				sum_value1 += pow_a * cosValue;
				pow_a = pow_a * a;
				pow_b = pow_b * b;
			}
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += sum_value1;
		}
	}

	real tmp_fitness_value = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];

	if (threadIdx.x == 0)
	{
		real sum_value2 = 0;
		pow_a = 1.0;
		pow_b = 1.0;
		for (int i = 0; i <= kmax; i++)
		{
			fix_value = 2 * M_PI * pow_b;
			sincos(fix_value * 0.5, &sinValue, &cosValue);
			sum_value2 += pow_a * cosValue;
			pow_a = pow_a * a;
			pow_b = pow_b * b;
		}
		tmp_fitness_value = tmp_fitness_value - calcuate_dim * sum_value2;
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
};
__global__ void global_f7(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	real sinValue = 0, cosValue = 0;

	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;
			tmp_value = tmp_value * tmp_value;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value;
		}
	}

	real tmp_fitness_value1 = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value1 = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];

	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 1;
	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;

		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;
			sincos(tmp_value / sqrt(index_dim + 1.0), &sinValue, &cosValue);

			tmp_value = cosValue;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] *= tmp_value;
		}
	}

	real tmp_fitness_value2 = device_ParallelMultiple(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value2 = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];


	if (threadIdx.x == 0)
	{
		d_fitness_value[ind_individual] = wi * (tmp_fitness_value1 / 4000.0 - tmp_fitness_value2 + 1.0) + bias;
	}
};
__global__ void global_f8(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;
	real sinValue = 0, cosValue = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;
			sincos(2 * M_PI * tmp_value, &sinValue, &cosValue);
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value * tmp_value - 10 * cosValue + 10;
		}
	}

	real tmp_fitness_value = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];

	if (threadIdx.x == 0)
	{
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
};
__global__ void global_f9(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value1, tmp_value2;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	real sinValue = 0, cosValue = 0;
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		if (index_dim < calcuate_dim)
		{
			tmp_value1 = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value1 = tmp_value1 * a_rate + shift + 4.209687462275036e+002;
			if (tmp_value1 <= 500 && tmp_value1 >= -500)
			{
				tmp_value2 = sqrt(abs(tmp_value1));
				sincos(tmp_value2, &sinValue, &cosValue);
				tmp_value2 = sinValue * tmp_value1;
			}
			else if (tmp_value1 < -500)
			{
#ifdef GPU_DOUBLE_PRECISION
				tmp_value2 = -500 + fmod(abs(tmp_value1), 500.0);
#endif
#ifdef GPU_SINGLE_PRECISION
				tmp_value2 = -500 + fmodf(abs(tmp_value1), 500.0);
#endif
				sincos(sqrt(fabs(tmp_value2)), &sinValue, &cosValue);
				tmp_value2 = tmp_value2 * sinValue - (tmp_value1 + 500) * (tmp_value1 + 500) / (10000.0 * calcuate_dim);
			}
			else
			{
#ifdef GPU_DOUBLE_PRECISION
				tmp_value2 = 500 - fmod(tmp_value1, 500.0);
#endif
#ifdef GPU_SINGLE_PRECISION
				tmp_value2 = 500 - fmodf(tmp_value1, 500.0);
#endif
				sincos(sqrt(fabs(tmp_value2)), &sinValue, &cosValue);
				real mp = tmp_value2 * sinValue;
				tmp_value2 = tmp_value2 * sinValue - (tmp_value1 - 500) * (tmp_value1 - 500) / (10000.0 * calcuate_dim);
			}
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value2;
		}
	}

	real tmp_fitness_value = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];


	if (threadIdx.x == 0)
	{
		tmp_fitness_value = 4.189828872724338e+002 * calcuate_dim - tmp_fitness_value;
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
};
__global__ void global_f10(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;

	int kmax = 32;
	real fix_value1 = 0;
	real fix_value2 = 10.0 / (pow(calcuate_dim + 0.0, 1.2));
	real fix_value3 = 10.0 / (pow(calcuate_dim + 0.0, 2.0));

	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 1;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;

		if (index_dim < calcuate_dim)
		{
			tmp_value = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value = tmp_value * a_rate + shift;

			real sum_value1 = 0;

			for (int j = 1; j < kmax; j++)
			{
				fix_value1 = pow(2.0, j + 0.0);
				tmp_value = tmp_value * fix_value1;
				tmp_value = abs(tmp_value - floor(tmp_value + 0.5));
				tmp_value = tmp_value / fix_value1;
				sum_value1 += tmp_value;
			}
#ifdef GPU_DOUBLE_PRECISION
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] *= pow(sum_value1 * (index_dim + 1.0) + 1.0, fix_value2);
#endif
#ifdef GPU_SINGLE_PRECISION
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] *= powf(sum_value1 * (index_dim + 1.0) + 1.0, fix_value2);
#endif
		}
	}

	real tmp_fitness_value = device_ParallelMultiple(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];

	if (threadIdx.x == 0)
	{
		tmp_fitness_value = fix_value3 * tmp_fitness_value - fix_value3;
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
};
__global__ void global_f11(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value1, tmp_value2;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	real fix_value1 = 0;
	real tmp_fitness_value1, tmp_fitness_value2, tmp_fitness_value3 = 0;
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		if (index_dim < calcuate_dim)
		{
			tmp_value1 = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value1 = tmp_value1 * a_rate + shift;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value1 * tmp_value1;
		}
	}

	tmp_fitness_value1 = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value1 = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];
	if (threadIdx.x == 0)
	{
		tmp_fitness_value2 = tmp_fitness_value1 - calcuate_dim;
		tmp_fitness_value2 = sqrt(sqrt(fabs(tmp_fitness_value2)));
		tmp_fitness_value1 *= 0.5;
	}

	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;

		if (index_dim < calcuate_dim)
		{
			tmp_value1 = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value1 = tmp_value1 * a_rate + shift;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value1;
		}
	}
	tmp_fitness_value3 = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value3 = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];

	if (threadIdx.x == 0)
	{
		tmp_fitness_value3 += tmp_fitness_value1;
		tmp_fitness_value3 = tmp_fitness_value3 / (calcuate_dim + 0.0) + 0.5;
		d_fitness_value[ind_individual] = wi * tmp_fitness_value2 + wi * tmp_fitness_value3 + bias;
	}
};
__global__ void global_f12(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value1, tmp_value2;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;

	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;
	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		if (index_dim < calcuate_dim)
		{
			tmp_value1 = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value1 = tmp_value1 * a_rate + shift;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value1 * tmp_value1;
		}
	}
	real tmp_fitness_value1 = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value1 = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];

	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;
	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		if (index_dim < calcuate_dim)
		{
			tmp_value1 = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			tmp_value1 = tmp_value1 * a_rate + shift;
			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value1;
		}
	}

	real tmp_fitness_value2 = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value2 = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];


	if (threadIdx.x == 0)
	{
		real tmp_fitness_value3 = 0;
		tmp_value2 = tmp_fitness_value1 * tmp_fitness_value1 - tmp_fitness_value2 * tmp_fitness_value2;
		if (tmp_value2 >= 0)
			tmp_fitness_value3 = sqrt(tmp_value2);
		else
			tmp_fitness_value3 = sqrt(-tmp_value2);

		real tmp_fitness_value4 = (0.5 * tmp_fitness_value1 + tmp_fitness_value2) / (calcuate_dim + 0.0) + 0.5;
		d_fitness_value[ind_individual] = wi * tmp_fitness_value3 + wi * tmp_fitness_value4 + bias;
	}
};
__global__ void global_f13(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value1, tmp_value2;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;

	for (int i = 0; i < loop_times; i++)
	{
		index_dim = threadIdx.x + i * blockDim.x;
		if (index_dim < calcuate_dim)
		{
			tmp_value1 = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
			if (index_dim < calcuate_dim - 1)
				tmp_value2 = d_elements[index_dim + 1 + ind_individual * next_pow2_problem_dim];
			else
				tmp_value2 = d_elements[0 + ind_individual * next_pow2_problem_dim];

			tmp_value1 = tmp_value1 * a_rate + shift;
			tmp_value2 = tmp_value2 * a_rate + shift;

			tmp_value2 = 100 * (tmp_value1 * tmp_value1 - tmp_value2) * (tmp_value1 * tmp_value1 - tmp_value2);
			tmp_value2 += (tmp_value1 - 1.0) * (tmp_value1 - 1.0);

			sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value2 * tmp_value2 / 4000.0 - cos(tmp_value2) + 1.0;
		}
	}

	real tmp_fitness_value = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
	tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];


	if (threadIdx.x == 0)
	{
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
}
__global__ void global_f14(real* d_fitness_value, real * d_elements,  int problem_dim, int calcuate_dim, real shift, real a_rate, real wi, real bias)
{
	real tmp_value1, tmp_value2, tmp_value3;
	int next_pow2_problem_dim = device_NextPow2Dim(problem_dim);
	int next_pow2_calcuate_dim = device_NextPow2Dim(calcuate_dim);
	int loop_times = next_pow2_calcuate_dim / blockDim.x;
	int ind_individual = threadIdx.y + blockIdx.y * blockDim.y;
	int index_dim = 0;
	sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;
	real sinValue = 0, cosValue = 0;
	real tmp_fitness_value = 0;
	if(calcuate_dim > 1)
	{
		for (int i = 0; i < loop_times; i++)
		{
			index_dim = threadIdx.x + i * blockDim.x;
			if (index_dim < calcuate_dim)
			{
				tmp_value1 = d_elements[index_dim + ind_individual * next_pow2_problem_dim];
				if (index_dim < calcuate_dim - 1)
					tmp_value2 = d_elements[index_dim + 1 + ind_individual * next_pow2_problem_dim];
				else
					tmp_value2 = d_elements[0 + ind_individual * next_pow2_problem_dim];

				tmp_value1 = tmp_value1 * a_rate + shift;
				tmp_value2 = tmp_value2 * a_rate + shift;

				tmp_value3 = tmp_value1 * tmp_value1 + tmp_value2 * tmp_value2;
				sincos(sqrt(tmp_value3), &sinValue, &cosValue);

				tmp_value3 = 0.5 + (sinValue * sinValue - 0.5) / ((1 + 0.001 * tmp_value3) * (1 + 0.001 * tmp_value3));
				sh_mem_CEC2014[threadIdx.x + threadIdx.y * WARP_SIZE] += tmp_value3;
			}
		}
		tmp_fitness_value = device_ParallelSum(sh_mem_CEC2014 + threadIdx.y * WARP_SIZE);
		tmp_fitness_value = sh_mem_CEC2014[threadIdx.y * WARP_SIZE];
	}
	else
	{
		if(threadIdx.x == 0)
		{
			tmp_value1 = d_elements[ind_individual * next_pow2_problem_dim];
			tmp_value1 = tmp_value1 * a_rate + shift;
			tmp_value3 = tmp_value1 * tmp_value1;
			sincos(sqrt(tmp_value3), &sinValue, &cosValue);

			tmp_value3 = 0.5 + (sinValue * sinValue - 0.5) / ((1 + 0.001 * tmp_value3) * (1 + 0.001 * tmp_value3));
			tmp_fitness_value = tmp_value3;
		}
	}


	if (threadIdx.x == 0)
	{
		d_fitness_value[ind_individual] = wi * tmp_fitness_value + bias;
	}
}
int f1(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);

	global_f1 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f2(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f2 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f3(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
		if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f3 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f4(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f4 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f5(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f5 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f6(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f6 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f7(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f7 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f8(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f8 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f9(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f9 << <grid, block, shared_mem_size>> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f10(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f10 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f11(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);

	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f11 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f12(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f12 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f13(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f13 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}
int f14(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, \
	int pop_size, int problem_dim, real shift, real a_rate, real wi, real bias, int shift_flag, int rotate_flag)
{
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	if (rotate_flag == 1)
		ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, shift_flag);
	else
		Shift(d_rotated_elements, d_original_elements, d_shift_data, problem_dim, pop_size, shift_flag);
	global_f14 << <grid, block, shared_mem_size >> >(d_fitness_value, d_rotated_elements, problem_dim, problem_dim, shift, a_rate, wi, bias);
	return 0;
}

int F1(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f1(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 1, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F2(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f2(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 1, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F3(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f3(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 1, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F4(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f4(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 1, 0.02048, 1, 0, SHIFT, ROTATE);
	return 0;

}

int F5(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f5(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 1, 1, 0, SHIFT, ROTATE);
	return 0;

}

int F6(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f6(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 0.005, 1, 0, SHIFT, ROTATE);

	return 0;
}

int F7(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f7(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 6.0, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F8(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f8(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 0.0512, 1, 0, SHIFT, NO_ROTATE);
	return 0;
}

int F9(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f8(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 0.0512, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F10(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f9(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 10.0, 1, 0, SHIFT, NO_ROTATE);
	return 0;
}

int F11(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f9(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 10.0, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F12(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f10(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 0.05, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F13(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f11(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, -1, 0.05, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F14(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f12(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, -1, 0.05, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F15(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f13(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 1, 0.05, 1, 0, SHIFT, ROTATE);
	return 0;
}

int F16(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, real * d_shift_data, real * d_rotated_data,  int pop_size, int problem_dim)
{
	f14(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim, 0, 1, 1, 0, SHIFT, ROTATE);
	return 0;
}

int HybirdFunc1(real* d_fitness_value, real * d_shuffled_elements, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, int * d_shuffle_data,  real * d_tmp_fitness, real *d_wi, \
	int pop_size, int problem_dim, real wi, real bias)
{
	real p[3] = { 0.3, 0.3, 0.4 };
	int dim_local[3];
	int num_func = 3;
	dim_local[0] = ceil(p[0] * problem_dim);
	dim_local[1] = ceil(p[1] * problem_dim);
	dim_local[2] = problem_dim - dim_local[0] - dim_local[1];
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, 1);
	Shuffle(d_shuffled_elements, d_rotated_elements, d_shuffle_data, problem_dim, pop_size);

	int accumulate_dim = 0;
	global_f9 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_shuffled_elements, problem_dim, dim_local[0], 0, 10.0, 1, 0);

	accumulate_dim += dim_local[0];
	global_f8 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[1], 0, 5.12 / 100.0, 1, 0);

	accumulate_dim += dim_local[1];
	global_f1 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[2], 0, 1, 1, 0);

	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, wi, bias);

	real tmp[1];
	cudaMemcpy(tmp, d_fitness_value, sizeof(real), cudaMemcpyDeviceToHost);
	return 0;

}
int HybirdFunc2(real* d_fitness_value, real * d_shuffled_elements, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, int * d_shuffle_data,  real * d_tmp_fitness, real *d_wi, \
	int pop_size, int problem_dim, real wi, real bias)
{
	real p[3] = { 0.3, 0.3, 0.4 };
	int dim_local[3];
	int num_func = 3;

	dim_local[0] = ceil(p[0] * problem_dim);
	dim_local[1] = ceil(p[1] * problem_dim);
	dim_local[2] = problem_dim - dim_local[0] - dim_local[1];
	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, 1);
	Shuffle(d_shuffled_elements, d_rotated_elements, d_shuffle_data, problem_dim, pop_size);

	int accumulate_dim = 0;
	global_f2 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_shuffled_elements, problem_dim, dim_local[0], 0, 1, 1, 0);

	accumulate_dim += dim_local[0];
	global_f12 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[1], -1, 5.0 / 100.0, 1, 0);

	accumulate_dim += dim_local[1];
	global_f8 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[2], 0, 5.12 / 100.0, 1, 0);

	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, wi, bias);


	return 0;
}
int HybirdFunc3(real* d_fitness_value, real * d_shuffled_elements, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, int * d_shuffle_data,  real * d_tmp_fitness, real *d_wi, \
	int pop_size, int problem_dim, real wi, real bias)
{
	real p[4] = { 0.2, 0.2, 0.3, 0.3 };
	int dim_local[4];
	int num_func = 4;

	dim_local[0] = ceil(p[0] * problem_dim);
	dim_local[1] = ceil(p[1] * problem_dim);
	dim_local[2] = ceil(p[2] * problem_dim);
	dim_local[3] = problem_dim - dim_local[0] - dim_local[1] - dim_local[2];
	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, 1);
	Shuffle(d_shuffled_elements, d_rotated_elements, d_shuffle_data, problem_dim, pop_size);

	int accumulate_dim = 0;
	global_f7 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_shuffled_elements, problem_dim, dim_local[0], 0, 6, 1, 0);

	accumulate_dim += dim_local[0];
	global_f6 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[1], 0, 0.005, 1, 0);

	accumulate_dim += dim_local[1];
	global_f4 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[2], 1, 0.02048, 1, 0);

	accumulate_dim += dim_local[2];
	global_f14 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 3 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[3], 0, 1, 1, 0);

	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, wi, bias);

	return 0;
}

int HybirdFunc4(real* d_fitness_value, real * d_shuffled_elements, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, int * d_shuffle_data,  real * d_tmp_fitness, real *d_wi, \
	int pop_size, int problem_dim, real wi, real bias)
{
	real p[4] = { 0.2, 0.2, 0.3, 0.3 };
	int dim_local[4];
	int num_func = 4;

	dim_local[0] = ceil(p[0] * problem_dim);
	dim_local[1] = ceil(p[1] * problem_dim);
	dim_local[2] = ceil(p[2] * problem_dim);
	dim_local[3] = problem_dim - dim_local[0] - dim_local[1] - dim_local[2];
	int next_pow2_dim = device_NextPow2Dim(problem_dim);

	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, 1);
	Shuffle(d_shuffled_elements, d_rotated_elements, d_shuffle_data, problem_dim, pop_size);

	int accumulate_dim = 0;
	global_f12 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_shuffled_elements, problem_dim, dim_local[0], -1, 5 / 100.0, 1, 0);

	accumulate_dim += dim_local[0];
	global_f3 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[1], 0, 1, 1, 0);

	accumulate_dim += dim_local[1];
	global_f13 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[2], 1, 5.0 / 100.0, 1, 0);

	accumulate_dim += dim_local[2];
	global_f8 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 3 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[3], 0, 5.12 / 100.0, 1, 0);
	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, wi, bias);

	return 0;
}

int HybirdFunc5(real* d_fitness_value, real * d_shuffled_elements, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, int * d_shuffle_data,  real * d_tmp_fitness, real *d_wi, \
	int pop_size, int problem_dim, real wi, real bias)
{
	real p[5] = { 0.1, 0.2, 0.2, 0.2, 0.3 };
	int dim_local[5];
	int num_func = 5;

	dim_local[0] = ceil(p[0] * problem_dim);
	dim_local[1] = ceil(p[1] * problem_dim);
	dim_local[2] = ceil(p[2] * problem_dim);
	dim_local[3] = ceil(p[3] * problem_dim);
	dim_local[4] = problem_dim - dim_local[0] - dim_local[1] - dim_local[2] - dim_local[3];
	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, 1);
	Shuffle(d_shuffled_elements, d_rotated_elements, d_shuffle_data, problem_dim, pop_size);

	int accumulate_dim = 0;
	global_f14 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_shuffled_elements, problem_dim, dim_local[0], 0, 1, 1, 0);

	accumulate_dim += dim_local[0];
	global_f12 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[1], -1, 5.0 / 100.0, 1, 0);

	accumulate_dim += dim_local[1];
	global_f4 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[2], 1, 2.048 / 100.0, 1, 0);

	accumulate_dim += dim_local[2];
	global_f9 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 3 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[3], 0, 1000.0 / 100.0, 1, 0);

	accumulate_dim += dim_local[3];
	global_f1 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 4 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[4], 0, 1, 1, 0);
	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, wi, bias);
	return 0;
}

int HybirdFunc6(real* d_fitness_value, real * d_shuffled_elements, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data, int * d_shuffle_data,  real * d_tmp_fitness, real *d_wi, \
	int pop_size, int problem_dim, real wi, real bias)
{
	real p[5] = { 0.1, 0.2, 0.2, 0.2, 0.3 };
	int dim_local[5];
	int num_func = 5;

	dim_local[0] = ceil(p[0] * problem_dim);
	dim_local[1] = ceil(p[1] * problem_dim);
	dim_local[2] = ceil(p[2] * problem_dim);
	dim_local[3] = ceil(p[3] * problem_dim);
	dim_local[4] = problem_dim - dim_local[0] - dim_local[1] - dim_local[2] - dim_local[3];
	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, problem_dim, pop_size, 1);
	Shuffle(d_shuffled_elements, d_rotated_elements, d_shuffle_data, problem_dim, pop_size);

	int accumulate_dim = 0;
	global_f10 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_shuffled_elements, problem_dim, dim_local[0], 0, 0.05, 1, 0);

	accumulate_dim += dim_local[0];
	global_f11 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[1], -1, 5.0 / 100.0, 1, 0);

	accumulate_dim += dim_local[1];
	global_f13 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[2], 1, 5.0 / 100.0, 1, 0);

	accumulate_dim += dim_local[2];
	global_f9 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 3 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[3], 0, 1000.0 / 100.0, 1, 0);

	accumulate_dim += dim_local[3];
	global_f5 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 4 * pop_size, d_shuffled_elements + accumulate_dim, problem_dim, dim_local[4], 0, 1, 1, 0);

	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, wi, bias);
	return 0;
}
int CompFunc1(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data,  real * d_tmp_fitness, real * d_wi, real *d_sigma, \
	int pop_size, int problem_dim, real bias)
{
	int num_func = 5;

	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);

	int offset = 0;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data, problem_dim, pop_size, 1);
	global_f4 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_rotated_elements, problem_dim, problem_dim, 1, 2.048 / 100.0, 1, 0);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f1 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1, 1e-6, 100);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f2 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1, 1e-26, 200);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f3 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 3 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1, 1e-6, 300);

	offset += next_pow2_dim;
	Shift(d_rotated_elements, d_original_elements, d_shift_data + offset, problem_dim, pop_size, 1);
	global_f1 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 4 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1, 1e-6, 400);

	CalW(d_wi, d_original_elements, d_shift_data, d_sigma, problem_dim, pop_size, num_func);
	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, 1, bias);

	return 0;
}

int CompFunc2(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data,  real * d_tmp_fitness, real * d_wi, real *d_sigma, \
	int pop_size, int problem_dim, real bias)
{
	int num_func = 3;
	int next_pow2_dim = device_NextPow2Dim(problem_dim);

	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	int offset = 0;
	Shift(d_rotated_elements, d_original_elements, d_shift_data + offset, problem_dim, pop_size, 1);
	global_f9 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_rotated_elements, problem_dim, problem_dim, 0, 1000.0 / 100.0, 1, 0);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f8 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 5.12 / 100.0, 1, 100);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f12 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_rotated_elements, problem_dim, problem_dim, -1, 5 / 100.0, 1, 200);

	CalW(d_wi, d_original_elements, d_shift_data, d_sigma, problem_dim, pop_size, num_func);
	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, 1, bias);

	return 0;
}

int CompFunc3(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data,  real * d_tmp_fitness, real * d_wi, real *d_sigma, \
	int pop_size, int problem_dim, real bias)
{
	int num_func = 3;
	int next_pow2_dim = device_NextPow2Dim(problem_dim);

	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);

	int offset = 0;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);

	global_f9 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_rotated_elements, problem_dim, problem_dim, 0, 1000.0 / 100.0, 0.25, 0);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f8 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 5.12 / 100.0, 1, 100);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f1 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1, 1e-7, 200);

	CalW(d_wi, d_original_elements, d_shift_data, d_sigma, problem_dim, pop_size, num_func);
	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, 1, bias);

	return 0;
}

int CompFunc4(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data,  real * d_tmp_fitness, real * d_wi, real *d_sigma, \
	int pop_size, int problem_dim, real bias)
{
	int num_func = 5;
	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	int offset = 0;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f9 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_rotated_elements, problem_dim, problem_dim, 0, 1000.0 / 100.0, 0.25, 0);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f11 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_rotated_elements, problem_dim, problem_dim, -1, 5.0 / 100.0, 1, 100);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f1 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1, 1e-7, 200);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f6 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 3 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 0.5 / 100.0, 2.5, 300);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f7 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 4 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 600.0 / 100.0, 10, 400);

	CalW(d_wi, d_original_elements, d_shift_data, d_sigma, problem_dim, pop_size, num_func);
	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, 1, bias);
	return 0;
}

int CompFunc5(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data,  real * d_tmp_fitness, real * d_wi, real *d_sigma, \
	int pop_size, int problem_dim, real bias)
{
	int num_func = 5;
	int next_pow2_dim = device_NextPow2Dim(problem_dim);

	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	int offset = 0;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f12 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_rotated_elements, problem_dim, problem_dim, -1, 5 / 100.0, 10, 0);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f8 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 5.12 / 100.0, 10, 100);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f9 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1000.0 / 100.0, 2.5, 200);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f6 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 3 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 0.5 / 100.0, 25, 300);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f1 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 4 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1, 1e-6, 400);

	CalW(d_wi, d_original_elements, d_shift_data, d_sigma, problem_dim, pop_size, num_func);
	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, 1, bias);

	return 0;
}

int CompFunc6(real* d_fitness_value, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data,  real * d_tmp_fitness, real * d_wi, real *d_sigma, \
	int pop_size, int problem_dim, real bias)
{
	int num_func = 5;
	int next_pow2_dim = device_NextPow2Dim(problem_dim);

	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	int offset = 0;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f13 << <grid, block, shared_mem_size >> >(d_tmp_fitness, d_rotated_elements, problem_dim, problem_dim, 1, 5.0 / 100.0, 2.5, 0);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f11 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 1 * pop_size, d_rotated_elements, problem_dim, problem_dim, -1, 5 / 100.0, 10, 100);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f9 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 2 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1000.0 / 100.0, 2.5, 200);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f14 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 3 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1, 5e-4, 300);

	offset += next_pow2_dim;
	ShiftAndRotation(d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, problem_dim, pop_size, 1);
	global_f1 << <grid, block, shared_mem_size >> >(d_tmp_fitness + 4 * pop_size, d_rotated_elements, problem_dim, problem_dim, 0, 1, 1e-6, 400);

	CalW(d_wi, d_original_elements, d_shift_data, d_sigma, problem_dim, pop_size, num_func);
	CombineFitness(d_fitness_value, d_tmp_fitness, num_func, d_wi, pop_size, 1, bias);

	return 0;
}

int CompFunc7(real* d_fitness_value, real* d_shuffled_elements, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data,  int * d_shuffle_data, real * d_tmp_fitness, real * d_wi, real * d_sigma, \
	int pop_size, int problem_dim, real bias)
{
	int num_func = 3;
	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	int max_hybrid_function_num = 4;
	int offset = 0, offset1 = pop_size * max_hybrid_function_num;
	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);

	offset = 0;
	HybirdFunc1(d_tmp_fitness + offset1, d_shuffled_elements, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, pop_size, problem_dim, 1, 0);

	offset += next_pow2_dim;
	HybirdFunc2(d_tmp_fitness + pop_size + offset1, d_shuffled_elements, d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, d_shuffle_data + offset, d_tmp_fitness, d_wi, pop_size, problem_dim, 1, 100);

	offset += next_pow2_dim;
	HybirdFunc3(d_tmp_fitness + 2 * pop_size + offset1, d_shuffled_elements, d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, d_shuffle_data + offset, d_tmp_fitness, d_wi, pop_size, problem_dim, 1, 200);


	CalW(d_wi + max_hybrid_function_num * pop_size, d_original_elements, d_shift_data, d_sigma, problem_dim, pop_size, num_func);
	CombineFitness(d_fitness_value, d_tmp_fitness + offset1, num_func, d_wi + max_hybrid_function_num * pop_size, pop_size, 1, bias);

	return 0;
}

int CompFunc8(real* d_fitness_value, real* d_shuffled_elements, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data,  int * d_shuffle_data, real * d_tmp_fitness, real * d_wi, real * d_sigma, \
	int pop_size, int problem_dim, real bias)
{
	int num_func = 3;
	int next_pow2_dim = device_NextPow2Dim(problem_dim);
	int max_hybrid_function_num = 5;

	dim3 grid, block;
	int shared_mem_size = 0;
	Configuration(&grid, &block, &shared_mem_size, pop_size);
	int offset = 0, offset1 = pop_size * max_hybrid_function_num;

	HybirdFunc4(d_tmp_fitness + offset1, d_shuffled_elements, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, pop_size, problem_dim, 1, 0);

	offset += next_pow2_dim;
	HybirdFunc5(d_tmp_fitness + pop_size + offset1, d_shuffled_elements, d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, d_shuffle_data + offset, d_tmp_fitness, d_wi, pop_size, problem_dim, 1, 100);

	offset += next_pow2_dim;
	HybirdFunc6(d_tmp_fitness + 2 * pop_size + offset1, d_shuffled_elements, d_rotated_elements, d_original_elements, d_shift_data + offset, d_rotated_data + offset * problem_dim, d_shuffle_data + offset, d_tmp_fitness, d_wi, pop_size, problem_dim, 1, 200);

	CalW(d_wi + max_hybrid_function_num * pop_size, d_original_elements, d_shift_data, d_sigma, problem_dim, pop_size, num_func);
	CombineFitness(d_fitness_value, d_tmp_fitness + offset1, num_func, d_wi + max_hybrid_function_num * pop_size, pop_size, 1, bias);

	return 0;
}

extern "C"
void API_evaluateFitness(real* d_fitness_value, real* d_shuffled_elements, real * d_rotated_elements, real * d_original_elements, \
	real * d_shift_data, real * d_rotated_data,  int * d_shuffle_data, real * d_tmp_fitness, real * d_wi, real * d_sigma, \
	int pop_size, int problem_dim, real bias, int function_ID)
{
	real wi = 1;
	switch (function_ID)
	{
	case(1) :
		F1(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(2) :
		F2(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(3) :
		F3(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(4) :
		F4(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(5) :
		F5(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(6) :
		F6(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(7) :
		F7(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(8) :
		F8(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(9) :
		F9(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(10) :
		F10(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(11) :
		F11(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(12) :
		F12(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(13) :
		F13(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(14) :
		F14(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(15) :
		F15(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(16) :
		F16(d_fitness_value, d_rotated_elements, d_original_elements, d_shift_data, d_rotated_data, pop_size, problem_dim);
		break;
	case(17) :
		HybirdFunc1(d_fitness_value, d_shuffled_elements, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, pop_size, problem_dim, wi, bias);
		break;
	case(18) :
		HybirdFunc2(d_fitness_value, d_shuffled_elements, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, pop_size, problem_dim, wi, bias);
		break;
	case(19) :
		HybirdFunc3(d_fitness_value, d_shuffled_elements, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, pop_size, problem_dim, wi, bias);
		break;
	case(20) :
		HybirdFunc4(d_fitness_value, d_shuffled_elements, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, pop_size, problem_dim, wi, bias);
		break;
	case(21) :
		HybirdFunc5(d_fitness_value, d_shuffled_elements, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, pop_size, problem_dim, wi, bias);
		break;
	case(22) :
		HybirdFunc6(d_fitness_value, d_shuffled_elements, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, pop_size, problem_dim, wi, bias);
		break;
	case(23) :
		CompFunc1(d_fitness_value, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_tmp_fitness, d_wi, d_sigma, \
		pop_size, problem_dim, bias);
		break;
	case(24) :
		CompFunc2(d_fitness_value, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_tmp_fitness, d_wi, d_sigma, \
		pop_size, problem_dim, bias);
		break;
	case(25) :
		CompFunc3(d_fitness_value, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_tmp_fitness, d_wi, d_sigma, \
		pop_size, problem_dim, bias);
		break;
	case(26) :
		CompFunc4(d_fitness_value, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_tmp_fitness, d_wi, d_sigma, \
		pop_size, problem_dim, bias);
		break;
	case(27) :
		CompFunc5(d_fitness_value, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_tmp_fitness, d_wi, d_sigma, \
		pop_size, problem_dim, bias);
		break;
	case(28) :
		CompFunc6(d_fitness_value, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_tmp_fitness, d_wi, d_sigma, \
		pop_size, problem_dim, bias);
		break;
	case(29) :
		CompFunc7(d_fitness_value, d_shuffled_elements, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, d_sigma, \
		pop_size, problem_dim, bias);
		break;
	case(30) :
		CompFunc8(d_fitness_value, d_shuffled_elements, d_rotated_elements, d_original_elements, \
		d_shift_data, d_rotated_data, d_shuffle_data, d_tmp_fitness, d_wi, d_sigma, \
		pop_size, problem_dim, bias);
		break;
	default:
		break;

	}
	cudaGetLastError();
};
