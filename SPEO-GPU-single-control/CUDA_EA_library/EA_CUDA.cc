#include "EA_CUDA.h"
EA_CUDA::EA_CUDA()
{

}

EA_CUDA::~EA_CUDA()
{

}

int EA_CUDA::Initilize(ProblemInfo problem_info, IslandInfo island_info)
{
    problem_info_ = problem_info;
    island_info_ = island_info;

    int next_pow2_dim = CalNextPow2Dim();
    int total_elements = island_info_.island_size * next_pow2_dim;

    h_population_.elements = new real[total_elements];
    h_population_.fitness_value = new real[island_info_.island_size];
    cudaMalloc(&d_population_.elements, total_elements * sizeof(real));
    cudaMemset(d_population_.elements, 0, total_elements * sizeof(real));
    cudaMalloc(&d_candidate_.elements, total_elements * sizeof(real));
    cudaMemset(d_candidate_.elements, 0, total_elements * sizeof(real));

    cudaMalloc(&d_population_.fitness_value, island_info_.island_size * sizeof(real));
    cudaMemset(d_population_.fitness_value, 0, island_info_.island_size * sizeof(real));
    cudaMalloc(&d_candidate_.fitness_value, island_info_.island_size* sizeof(real));
    cudaMemset(d_candidate_.fitness_value, 0, island_info_.island_size * sizeof(real));

    cudaMalloc((void **)&d_rand_states_, total_elements * sizeof(curandState));

    cudaMalloc(&d_island_info_.permutated_index, island_info_.island_size * sizeof(int));
    cudaMemset(d_island_info_.permutated_index, 0, island_info_.island_size * sizeof(int));
    d_island_info_.island_size = island_info_.island_size;
    d_island_info_.subisland_num = island_info_.subisland_num;

    cec2014_.Initilize(problem_info_.function_ID, problem_info_.dim);

    API_Initilize(d_population_, d_rand_states_, d_island_info_, problem_info_);
    cec2014_cuda_.Initilize(problem_info_.function_ID, island_info_.island_size, problem_info_.dim);
    cec2014_cuda_.EvaluateFitness(d_population_.fitness_value, d_population_.elements);
    cudaMemcpy(d_candidate_.elements, d_population_.elements, total_elements * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_candidate_.fitness_value, d_population_.fitness_value, island_info_.island_size * sizeof(real), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    vector<int> permutated_index;///= random_.Permutate(island_info_.island_size, island_info_.island_size);

    for(int i = 0; i < island_info_.island_size; i++)
        permutated_index.push_back(i);
    RegroupIslands(permutated_index, island_info_);
    return 0;
}

int EA_CUDA::Unitilize()
{
    cec2014_.Unitilize();
    delete []h_population_.elements;
    delete []h_population_.fitness_value;
    cec2014_cuda_.Unitilize();
    cudaFree(d_population_.elements);
    cudaFree(d_candidate_.elements);
    cudaFree(d_population_.fitness_value);
    cudaFree(d_candidate_.fitness_value);
    cudaFree(d_rand_states_);
    cudaFree(d_island_info_.permutated_index);
    return 0;
}

int EA_CUDA::InitilizePopulation(Population & population)
{
    for(int i = 0; i < island_info_.island_size; i++)
    {
        Individual tmp_individual;
        for (int j = 0; j < problem_info_.dim; j++)
            tmp_individual.elements.push_back(random_.RandRealUnif(problem_info_.min_bound, problem_info_.max_bound));

        tmp_individual.fitness_value = 1E20;
        population.push_back(tmp_individual);
    }
#ifndef EA_CPU
	TransferDataToCPU(population);
#endif
    return 0;
}

int EA_CUDA::RegroupIslands(vector<int> &permutate_index, IslandInfo island_info)
{
    GPU_IslandInfo h_island_info;
    h_island_info.permutated_index = new int[island_info_.island_size];

    for (int i = 0; i < island_info_.island_size; i++)
        h_island_info.permutated_index[i] = permutate_index[i];
    island_info_ = island_info;
    d_island_info_.subisland_num = island_info_.subisland_num;
    cudaMemcpy(d_island_info_.permutated_index, h_island_info.permutated_index, sizeof(int) * island_info_.island_size, cudaMemcpyHostToDevice);

    API_CheckCUDAError();
    delete [] h_island_info.permutated_index;
    return 0;
}


Individual EA_CUDA::FindBestIndividual(Population & population)
{
#ifndef EA_CPU
    TransferDataToCPU(population);
#endif
    int best_individual_ind = 0;
    double best_individual_fitness_value = population[0].fitness_value;
    for(int i = 1; i < population.size(); i++)
    {
        if(population[i].fitness_value < best_individual_fitness_value)
        {
            best_individual_ind = i;
            best_individual_fitness_value = population[i].fitness_value;
        }
    }
	return population[best_individual_ind];
}

int EA_CUDA::TransferDataFromCPU(Population &population)
{
    int next_pow2_dim = CalNextPow2Dim();
	int total_elements = island_info_.island_size * next_pow2_dim;

	memset(h_population_.elements, 0, sizeof(real) * total_elements);
	memset(h_population_.fitness_value, 0, sizeof(real) * island_info_.island_size);

    int count = 0;
    // for(int i = 0; i < population.size(); i++)
    // {
    //     for (int j = 0; j < next_pow2_dim; j++)
    //     {
    //         if(j < problem_info_.dim)
    //         {
    //             real tmp = (real) population[i].elements[j];
    //             h_population_.elements[count] = tmp;
    //         }
    //         count++;
    //     }
    //     real tmp = (real) population[i].fitness_value;
    //     h_population_.fitness_value[i] = tmp;
    // }


    for(int i = 0; i < population.size(); i++)
    {
        copy (population[i].elements.begin(), population[i].elements.end(), h_population_.elements + count);
        h_population_.fitness_value[i] = population[i].fitness_value;
        count += next_pow2_dim;
    }

	cudaMemcpy(d_population_.elements, h_population_.elements, sizeof(real) * total_elements, cudaMemcpyHostToDevice);
	cudaMemcpy(d_population_.fitness_value, h_population_.fitness_value, sizeof(real) * island_info_.island_size, cudaMemcpyHostToDevice);

    return 0;
};

int EA_CUDA::TransferDataToCPU(Population &population)
{
    int next_pow2_dim = CalNextPow2Dim();
	int total_elements = island_info_.island_size * next_pow2_dim;

	memset(h_population_.elements, 0, sizeof(real) * total_elements);
	memset(h_population_.fitness_value, 0, sizeof(real) * island_info_.island_size);
    cudaDeviceSynchronize();

	cudaMemcpy(h_population_.elements, d_population_.elements, sizeof(real) * total_elements, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_population_.fitness_value, d_population_.fitness_value, sizeof(real) * island_info_.island_size, cudaMemcpyDeviceToHost);

//    printf("%lf\t%lf\n", *h_population_.elements, *h_population_.fitness_value);
    int count = 0;
	for (int i = 0; i < island_info_.island_size; i++)
    {
        // for (int j = 0; j < next_pow2_dim; j++)
        // {
        //     if(j < problem_info_.dim)
        //     {
        //         real tmp = (real) h_population_.elements[count];
        //         population[i].elements[j] = tmp;
        //     }
        //     count++;
        // }
        // real tmp = (real) h_population_.fitness_value[i];
        // population[i].fitness_value = tmp;
        population[i].elements.assign(\
			h_population_.elements + count, h_population_.elements + problem_info_.dim + count);
        count += next_pow2_dim;
        population[i].fitness_value = h_population_.fitness_value[i];
    }

    return 0;
};


int EA_CUDA::CalNextPow2Dim()
{
    int next_pow2_dim = 1;
    while (next_pow2_dim < problem_info_.dim)
        next_pow2_dim <<= 1;
    if (next_pow2_dim < WARP_SIZE)
        next_pow2_dim = WARP_SIZE;

    return next_pow2_dim;
}

real EA_CUDA::CheckBound(real to_check_elements, real min_bound, real max_bound)
{
    while ((to_check_elements < min_bound) || (to_check_elements > max_bound))
    {
        if (to_check_elements < min_bound)
            to_check_elements = min_bound + (min_bound - to_check_elements);
        if (to_check_elements > max_bound)
            to_check_elements = max_bound - (to_check_elements - max_bound);
    }
    return to_check_elements;
}
