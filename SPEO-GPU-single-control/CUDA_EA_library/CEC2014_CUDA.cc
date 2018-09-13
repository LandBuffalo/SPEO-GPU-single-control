#include "CEC2014_CUDA.h"

struct myclass
{
	int ID;
	real distance;
};
bool comparator ( const myclass& l, const myclass& r)
{ 
   	return l.distance < r.distance; 
};
CEC2014_CUDA::CEC2014_CUDA()
{
	ID_func_ = 0;
	dim_ = 0;
	size_pop_ = 0;
}

CEC2014_CUDA::~CEC2014_CUDA()
{

}

int CEC2014_CUDA::Initilize(int ID_func, int size_pop, int dim)
{

	ID_func_ = ID_func;
	size_pop_ = size_pop;
	dim_ = dim;

	flag_composition_ = 1;
	CalConfigCEC2014();
	next_pow2_dim_ = 1;
	while (next_pow2_dim_ < dim) 
	next_pow2_dim_ <<= 1;
if(next_pow2_dim_ < WARP_SIZE)
	next_pow2_dim_ = WARP_SIZE;
	LoadData();
	MallocAndMemSet();
	return 0;
}

int CEC2014_CUDA::Unitilize()
{
	cudaFree(d_M_);
	cudaFree(d_shuffle_);
	cudaFree(d_pop_rotated_);
	cudaFree(d_pop_shifted_);
	cudaFree(d_pop_shuffled_);
	cudaFree(d_shift_);

	if(num_composition_func_ > 1 || num_hybrid_func_ > 1)
	{
		cudaFree(d_wi_);
		cudaFree(d_tmp_fitness_);
	}
	if(ID_func_ > 23)
	{
		cudaFree(d_sigma_);
	}

#ifdef DISTANCE
    global_optima_.clear();
#endif
	return 0;

}
int CEC2014_CUDA::MallocAndMemSet()
{
	cudaMalloc(&d_pop_rotated_, next_pow2_dim_ * size_pop_ * sizeof(real));
	cudaMalloc(&d_pop_shifted_, next_pow2_dim_ * size_pop_ * sizeof(real));
	cudaMalloc(&d_pop_shuffled_, next_pow2_dim_ * size_pop_ * sizeof(real));

	cudaMemset(d_pop_rotated_, 0, next_pow2_dim_ * size_pop_ * sizeof(real));
	cudaMemset(d_pop_shifted_, 0, next_pow2_dim_ * size_pop_ * sizeof(real));
	cudaMemset(d_pop_shuffled_, 0, next_pow2_dim_ * size_pop_ * sizeof(real));

	if(num_composition_func_ > 1 && num_hybrid_func_ == 1)
	{
		cudaMalloc(&d_tmp_fitness_, size_pop_ * num_composition_func_ * sizeof(real));
		cudaMalloc(&d_wi_, size_pop_ * num_composition_func_ * sizeof(real));

		cudaMemset(d_wi_, 0, size_pop_ * num_composition_func_ * sizeof(real));
		cudaMemset(d_tmp_fitness_, 0,size_pop_ * num_composition_func_ * sizeof(real));

		real *h_wi = new real [num_composition_func_ * size_pop_];
		for(int i = 0; i <  num_composition_func_; i++)
		{
			for(int j = 0; j < size_pop_; j++)
				h_wi[j + i * size_pop_] = 1;
		}
		cudaMemcpy(d_wi_, h_wi, num_composition_func_ * size_pop_ * sizeof(real), cudaMemcpyHostToDevice);
	}

	if(num_composition_func_ == 1 && num_hybrid_func_ > 1)
	{
		cudaMalloc(&d_tmp_fitness_, size_pop_ * num_hybrid_func_ * sizeof(real));
		cudaMemset(d_tmp_fitness_, 0,size_pop_ * num_hybrid_func_ * sizeof(real));
		cudaMalloc(&d_wi_, size_pop_ * num_hybrid_func_ * sizeof(real));
		cudaMemset(d_wi_, 0, size_pop_ * num_hybrid_func_ * sizeof(real));
		real *h_wi = new real [num_hybrid_func_ * size_pop_];
		for(int i = 0; i <  num_hybrid_func_; i++)
		{
			for(int j = 0; j < size_pop_; j++)
				h_wi[j + i * size_pop_] = 1;
		}
		cudaMemcpy(d_wi_, h_wi, num_hybrid_func_ * size_pop_ * sizeof(real), cudaMemcpyHostToDevice);
	}

	if(num_composition_func_ > 1 && num_hybrid_func_ > 1)
	{
		int length = num_hybrid_func_ + num_composition_func_;
		cudaMalloc(&d_tmp_fitness_, size_pop_ * length * sizeof(real));
		cudaMemset(d_tmp_fitness_, 0, size_pop_ * length * sizeof(real));

		cudaMalloc(&d_wi_, size_pop_ * length * sizeof(real));
		cudaMemset(d_wi_, 0, size_pop_ * length * sizeof(real));

		real *h_wi = new real [length * size_pop_];
		for(int i = 0; i <  length; i++)
		{
			for(int j = 0; j < size_pop_; j++)
				h_wi[j + i * size_pop_] = 1;
		}
		cudaMemcpy(d_wi_, h_wi, length * size_pop_ * sizeof(real), cudaMemcpyHostToDevice);
	}
	return 0;

}
vector<real>	CEC2014_CUDA::DistanceFromGlobalOptima(Population &extreme_individual, Population &population)
{
	vector<real> distance;
	vector<myclass> myclass_distances;
	for(int i = 0; i < population.size(); i++)
	{
		real tmp = 0;
		myclass tmp_mtclass;
		for(int j = 0; j < dim_; j++)
			tmp += (population[i].elements[j] - global_optima_[j]) * (population[i].elements[j] - global_optima_[j]);
		tmp_mtclass.ID = i;
		tmp_mtclass.distance = sqrt(tmp);
		myclass_distances.push_back(tmp_mtclass);
	}
	sort(myclass_distances.begin(), myclass_distances.end(), comparator);

	for(int i = 0; i < population.size(); i++)
	{
		extreme_individual.push_back(population[myclass_distances[i].ID]);
		distance.push_back(myclass_distances[i].distance);
	}	

	return distance;

}
vector<real> 	CEC2014_CUDA::GlobalOptima()
{
	return global_optima_;
}

int CEC2014_CUDA::LoadData()
{

	cudaMalloc(&d_M_, next_pow2_dim_ * dim_ * num_composition_func_ * sizeof(real));
	cudaMalloc(&d_shuffle_, next_pow2_dim_ * num_composition_func_ * sizeof(int));
	cudaMalloc(&d_shift_, next_pow2_dim_ * num_composition_func_ * sizeof(real));

	cudaMemset(d_M_, 0, next_pow2_dim_ * dim_ * num_composition_func_ * sizeof(real));
	cudaMemset(d_shuffle_, 0, next_pow2_dim_ * num_composition_func_ * sizeof(int));
	cudaMemset(d_shift_, 0, next_pow2_dim_ * num_composition_func_ * sizeof(real));


	real *h_shift = new real[num_composition_func_ * next_pow2_dim_];
	real *h_M = new real[dim_ * num_composition_func_ * next_pow2_dim_];
	int *h_shuffle = new int[num_composition_func_ * next_pow2_dim_];
	real tmp;

	for (int i = 0; i < num_composition_func_; i++)
		for (int j = 0; j < next_pow2_dim_; j++)
		{
			h_shift[j + i * next_pow2_dim_] = 0;
			h_shuffle[j + i * next_pow2_dim_] = j;
		}
	for (int k = 0; k < num_composition_func_; k++)
		for (int i = 0; i < next_pow2_dim_; i++)
			for (int j = 0; j < dim_; j++)
			{
				if (i == j)
					h_M[j + i * dim_ + k * dim_ * next_pow2_dim_] = 1;
				else
					h_M[j + i * dim_ + k * dim_ * next_pow2_dim_] = 0;

			}
	char fileName[100];

	sprintf(fileName, "input_data/shift_data_%d.txt", ID_func_);
	FILE *file = fopen(fileName, "r");

	if (flag_composition_ == 1)
	{
		for (int i = 0; i < num_composition_func_; i++)
		{
			for (int j = 0; j < MAX_DIM; j++)
			{
				if(j < dim_)
				{
#ifdef GPU_DOUBLE_PRECISION
					fscanf(file, "%lf", &h_shift[j + i * next_pow2_dim_]);
#endif
#ifdef GPU_SINGLE_PRECISION
					fscanf(file, "%f", &h_shift[j + i * next_pow2_dim_]);
#endif
					if(i == 0)
						global_optima_.push_back(h_shift[j + i * next_pow2_dim_]);
				}	
				else
				{
#ifdef GPU_DOUBLE_PRECISION
					fscanf(file, "%lf", &tmp);
#endif
#ifdef GPU_SINGLE_PRECISION
					fscanf(file, "%f", &tmp);
#endif
				}

			}
		}

	}
	else
	{
		for (int i = 0; i < dim_; i++)
		{

#ifdef GPU_DOUBLE_PRECISION
			fscanf(file, "%lf", &h_shift[i]);
#endif
#ifdef GPU_SINGLE_PRECISION
			fscanf(file, "%f", &h_shift[i]);
#endif
		}
	}
	fclose(file);

	sprintf(fileName, "input_data/M_%d_D%d.txt", ID_func_, dim_);
	file = fopen(fileName, "r");
	for (int k = 0; k < num_composition_func_; k++)
	{
			for (int i = 0; i < next_pow2_dim_; i++)
			{
				if (i < dim_)
				{
					for (int j = 0; j < dim_; j++)
					{
#ifdef GPU_DOUBLE_PRECISION
						fscanf(file, "%lf", &h_M[i + (j + k * next_pow2_dim_) * dim_]);
#endif
#ifdef GPU_SINGLE_PRECISION
						fscanf(file, "%f", &h_M[i + (j + k * next_pow2_dim_) * dim_]);
#endif
					}

				}
				else
				{
					for (int j = 0; j < dim_; j++)
						h_M[j + (i + k * next_pow2_dim_) * dim_] = 0;
				}
			}

	}
	fclose(file);
	sprintf(fileName, "input_data/shuffle_data_%d_D%d.txt", ID_func_, dim_);
	file = fopen(fileName, "r");
	for (int i = 0; i < num_composition_func_; i++)
		for (int j = 0; j < dim_; j++)
			fscanf(file, "%d", &h_shuffle[j + i * next_pow2_dim_]);
	fclose(file);

	cudaMemcpy(d_shift_, h_shift, num_composition_func_ * next_pow2_dim_ * sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpy(d_M_, h_M, dim_ * num_composition_func_ * next_pow2_dim_ * sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpy(d_shuffle_, h_shuffle, num_composition_func_ * next_pow2_dim_ * sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_M;
	delete[] h_shift;
	delete[] h_shuffle;
	// //	HANDLE_CUDA_ERROR(cudaMemcpyFromSymbol(h_shift, d_shift, MAX_FUNC_COMPOSITION * MAX_DIM  * sizeof(real)));
	return 0;

}


int CEC2014_CUDA::EvaluateFitness(real * d_fval, real * d_pop)
{
	API_evaluateFitness(d_fval, d_pop_shuffled_, d_pop_rotated_, d_pop, \
				d_shift_, d_M_, d_shuffle_, d_tmp_fitness_,d_wi_,d_sigma_,\
				size_pop_, dim_, 0, ID_func_);
    return 0;
}
void CEC2014_CUDA::CalConfigCEC2014()
{
	if (ID_func_ <= 16)
	{
		num_composition_func_ = 1;
		flag_composition_ = 0;
		num_hybrid_func_ = 1;
	}
	else if(ID_func_ == 17)
	{
		num_composition_func_ = 1;
		num_hybrid_func_ = 3;	
	}
	else if(ID_func_ == 18)
	{
		num_composition_func_ = 1;
		num_hybrid_func_ = 3;	
	}
	else if(ID_func_ == 19)
	{
		num_composition_func_ = 1;
		num_hybrid_func_ = 4;	
	}
	else if(ID_func_ == 20)
	{
		num_composition_func_ = 1;
		num_hybrid_func_ = 4;	
	}
	else if(ID_func_ == 21)
	{
		num_composition_func_ = 1;
		num_hybrid_func_ = 5;	
	}
	else if(ID_func_ == 22)
	{
		num_composition_func_ = 1;
		num_hybrid_func_ = 5;	
	}
	else if(ID_func_ == 23)
	{
		num_composition_func_ = 5;
		num_hybrid_func_ = 1;	
		real h_sigma[5] = {10, 20, 30, 40, 50};
		cudaMalloc(&d_sigma_, num_composition_func_ * sizeof(real));
		cudaMemcpy(d_sigma_, h_sigma, num_composition_func_ * sizeof(real), cudaMemcpyHostToDevice);
	}
	else if(ID_func_ == 24)
	{
		num_composition_func_ = 3;
		num_hybrid_func_ = 1;
		real h_sigma[3] = {20, 20, 20};
		cudaMalloc(&d_sigma_, num_composition_func_ * sizeof(real));
		cudaMemcpy(d_sigma_, h_sigma, num_composition_func_ * sizeof(real), cudaMemcpyHostToDevice);
	}
	else if(ID_func_ == 25)
	{
		num_composition_func_ = 3;
		num_hybrid_func_ = 1;
		real h_sigma[3] = {10, 30, 50};
		cudaMalloc(&d_sigma_, num_composition_func_ * sizeof(real));
		cudaMemcpy(d_sigma_, h_sigma, num_composition_func_ * sizeof(real), cudaMemcpyHostToDevice);
	}	
	else if(ID_func_ == 26)
	{
		num_composition_func_ = 5;
		num_hybrid_func_ = 1;	
		real h_sigma[5] = {10, 10, 10, 10, 10};
		cudaMalloc(&d_sigma_, num_composition_func_ * sizeof(real));
		cudaMemcpy(d_sigma_, h_sigma, num_composition_func_ * sizeof(real), cudaMemcpyHostToDevice);
	}
	else if(ID_func_ == 27)
	{
		num_composition_func_ = 5;
		num_hybrid_func_ = 1;
		real h_sigma[5] = {10, 10, 10, 20, 20};
		cudaMalloc(&d_sigma_, num_composition_func_ * sizeof(real));
		cudaMemcpy(d_sigma_, h_sigma, num_composition_func_ * sizeof(real), cudaMemcpyHostToDevice);
	}
	else if(ID_func_ == 28)
	{
		num_composition_func_ = 5;
		num_hybrid_func_ = 1;
		real h_sigma[5] = {10, 20, 30, 40, 50};
		cudaMalloc(&d_sigma_, num_composition_func_ * sizeof(real));
		cudaMemcpy(d_sigma_, h_sigma, num_composition_func_ * sizeof(real), cudaMemcpyHostToDevice);
	}
	else if(ID_func_ == 29)
	{
		num_composition_func_ = 3;
		num_hybrid_func_ = 4;	
		real h_sigma[3] = {10, 30, 50};
		cudaMalloc(&d_sigma_, num_composition_func_ * sizeof(real));
		cudaMemcpy(d_sigma_, h_sigma, num_composition_func_ * sizeof(real), cudaMemcpyHostToDevice);
	}
	else if(ID_func_ == 30)
	{
		num_composition_func_ = 3;
		num_hybrid_func_ = 5;	
		real h_sigma[3] = {10, 30, 50};
		cudaMalloc(&d_sigma_, num_composition_func_ * sizeof(real));
		cudaMemcpy(d_sigma_, h_sigma, num_composition_func_ * sizeof(real), cudaMemcpyHostToDevice);
	}

}

