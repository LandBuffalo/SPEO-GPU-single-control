#include "EA_CUDA.h"
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

int CalConfigurations(dim3 & block_dim, dim3 & grid_dim, GPU_IslandInfo GPU_island_info)
{
	int total_individual_num = GPU_island_info.island_size;
	if (MAX_THREAD / WARP_SIZE < total_individual_num)
	{
		block_dim = dim3(WARP_SIZE, MAX_THREAD / WARP_SIZE, 1);
		grid_dim = dim3(total_individual_num / block_dim.y, 1, 1);
	}
	else
	{
		block_dim = dim3(WARP_SIZE, total_individual_num, 1);
		grid_dim = dim3(1, 1, 1);
	}
	return 0;
}



inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
		         file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif

	return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
		         file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
		         file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif

	return;
}

extern "C"
int API_CheckCUDAError()
{
	CudaCheckError();
	return 0;
}

#ifndef NEXT_POW2
#define NEXT_POW2
inline __device__ __host__ int NextPow2(int x)
{
	int y = 1;
	while (y < x) y <<= 1;
	if (y < WARP_SIZE)
		y = WARP_SIZE;
	return y;
}
#endif



__global__ void global_SetupRandomState(curandState * d_rand_states, int seed)
{
	int random_state_ID = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	curand_init(seed, random_state_ID, 0, &d_rand_states[random_state_ID]);
}

__global__ void global_InitiPopulation(GPU_Population d_population, curandState * d_rand_states, \
	ProblemInfo problem_info)
{
	int dim = problem_info.dim;

	int next_pow2_dim = NextPow2(dim);
	int loop_times = next_pow2_dim / WARP_SIZE;

	int individual_ID = threadIdx.y + blockIdx.x * blockDim.y;
	int random_state_ID = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;

	curandState local_state = d_rand_states[random_state_ID];

	for(int i = 0; i < loop_times; i++)
	{
		int element_ID = threadIdx.x + i * WARP_SIZE;
		if(element_ID < dim)
#ifdef GPU_DOUBLE_PRECISION
			d_population.elements[element_ID + individual_ID * next_pow2_dim] = problem_info.min_bound + curand_uniform_double(&local_state) * (problem_info.max_bound - problem_info.min_bound);
#endif
#ifdef GPU_SINGLE_PRECISION
			d_population.elements[element_ID + individual_ID * next_pow2_dim] = problem_info.min_bound + curand_uniform(&local_state) * (problem_info.max_bound - problem_info.min_bound);
#endif

    }

	d_rand_states[random_state_ID] = local_state;
}


extern "C"
int API_Initilize(GPU_Population d_population, curandState *d_rand_states, \
	GPU_IslandInfo GPU_island_info, ProblemInfo problem_info)
{
	dim3 block_dim, grid_dim;
	CalConfigurations(block_dim, grid_dim, GPU_island_info);

	global_SetupRandomState <<<grid_dim, block_dim>>>(d_rand_states, problem_info.seed);
	CudaCheckError();

	global_InitiPopulation <<< grid_dim, block_dim >>>(d_population, d_rand_states, problem_info);
	CudaCheckError();
	return 0;
}
