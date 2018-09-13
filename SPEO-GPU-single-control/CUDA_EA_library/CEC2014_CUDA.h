#ifndef __CEC2014_CUDA_H__
#define __CEC2014_CUDA_H__

#include <stdio.h>
#include <vector>
#include <fstream>

#include "cuda_runtime.h"
#include "math.h"
#include "config.h"
#include <algorithm>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_functions.h>
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795029
#endif

#ifndef M_E
#define M_E 2.7182818284590452353602874713526625
#endif
#define MAX_THREAD_PER_BLOCK 1024
#define MAX_NUM_COMP_FUNC 5
#define WARP_SIZE 32
#define TILE_WIDTH 16
#define MAX_DIM 100

#define NO_SHIFT 0
#define NO_ROTATE 0
#define SHIFT 1
#define ROTATE 1


extern "C" void API_evaluateFitness(real* d_fitness_value, real* d_shuffled_elements,  real * d_rotated_elements, real * d_original_elements, \
				real * d_shift_data, real * d_rotated_data, int * d_shuffle_data, real * d_tmp_fitness,real * d_wi,real * d_sigma,\
				int pop_size, int problem_dim, real bias, int function_ID);
class  CEC2014_CUDA
{
private:
	int						sigma_[MAX_NUM_COMP_FUNC];

	int 					flag_composition_;
	int						num_composition_func_;
	int						num_hybrid_func_;

	int						ID_func_;
	int						size_pop_;
	int						dim_;
	size_t					next_pow2_dim_;
	size_t					next_pow2_pop_;
	real					bias_;
	vector<real>			global_optima_;

	int *					d_shuffle_;
	real *				d_M_;
	real *				d_shift_;

	real *				d_pop_rotated_;
	real *				d_pop_shifted_;
	real *				d_pop_shuffled_;
	real *				d_wi_;
	real *				d_sigma_;
	real *				d_tmp_fitness_;


	void					ShiftRotate(real * d_pop);
	void					CalConfigCEC2014();
	int					    LoadData();
	int						MallocAndMemSet();
	

public:
							CEC2014_CUDA();
							~CEC2014_CUDA();
	vector<real>			GlobalOptima();
	int						Initilize(int ID_func, int size_pop, int dim);
	int						Unitilize();
	int					EvaluateFitness(real * d_fval, real * d_pop);
	vector<real>			DistanceFromGlobalOptima(Population &nearest_individual, Population &population);

};

#endif

