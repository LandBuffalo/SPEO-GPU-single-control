#include "EA_CUDA.h"
extern "C"
int API_DE_GenerateNewPopulation(GPU_Population d_candidate, GPU_Population d_population, int * d_best_individual_ID, GPU_IslandInfo d_island_info, DEInfo *d_DE_info, curandState *d_rand_states, ProblemInfo problem_info);
extern "C"
int API_DE_SelectSurvival(GPU_Population d_population, GPU_Population d_candidate, GPU_IslandInfo GPU_island_info, ProblemInfo problem_info);

DE_CUDA::DE_CUDA(NodeInfo node_info)
{
	node_info_ = node_info;
    cudaSetDevice(node_info_.GPU_ID % 2);
}

DE_CUDA::~DE_CUDA()
{

}

string DE_CUDA::GetParameters()
{

	string str;
	ostringstream temp1, temp2;
	string parameters = "CR/F=";

	double CR = DE_info_.CR;
	temp1<<CR;
	str=temp1.str();
	parameters.append(str);

	parameters.append("/");
	double F = DE_info_.F;
	temp2<<F;
	str=temp2.str();
	parameters.append(str);

	if(DE_info_.strategy_ID == 0)
		parameters.append("_current/1/bin");
	else if(DE_info_.strategy_ID == 1)
		parameters.append("_current/2/bin");
	else if(DE_info_.strategy_ID == 2)
		parameters.append("_current-to-best/1/bin");
	else if(DE_info_.strategy_ID == 3)
		parameters.append("_current-to-best/2/bin");
	else if(DE_info_.strategy_ID == 4)
		parameters.append("_rand/1/bin");
	else if(DE_info_.strategy_ID == 5)
		parameters.append("_rand/2/bin");
	else if(DE_info_.strategy_ID == 6)
		parameters.append("_best/1/bin");
	else if(DE_info_.strategy_ID == 7)
		parameters.append("_best/2/bin");
	else if(DE_info_.strategy_ID == 8)
		parameters.append("_current_to_rand/1/bin");
	return parameters;
}

int DE_CUDA::Initilize(ProblemInfo problem_info, IslandInfo island_info)
{
	EA_CUDA::Initilize(problem_info, island_info);

	DE_info_.CR = 0.9;
	DE_info_.F = 0.5;
	DE_info_.strategy_ID = 4;
	DE_info_.pop_size = -1;
	int max_subisland_size = island_info.island_size / MIN_SUBISLAND_SIZE;
	cudaMalloc(&d_DE_info_, max_subisland_size * sizeof(DEInfo));

	cudaMalloc(&d_best_individual_ID_,  max_subisland_size * sizeof(int));

	cudaMemset(d_best_individual_ID_, 0,  max_subisland_size * sizeof(int));
	

	ConfigureEA();
	
    return 0;
}

int DE_CUDA::Unitilize()
{

    cudaFree(d_DE_info_);
    cudaFree(d_best_individual_ID_);

	EA_CUDA::Unitilize();

    return 0;
}

int DE_CUDA::ConfigureEA()
{
	DEInfo *h_DE_info = new DEInfo[island_info_.subisland_num];
	if(island_info_.configure_EA == "dynamic")
	{
		double CR[3] = {0.1, 0.9};
		int strategy_ID[3] = {4, 6};
		for (int i = 0; i < island_info_.subisland_num; i++)
		{
			h_DE_info[i].CR = (real) CR[random_.RandIntUnif(0,1)];
			h_DE_info[i].F = (real) DE_info_.F;
			h_DE_info[i].strategy_ID = strategy_ID[random_.RandIntUnif(0,1)];
			h_DE_info[i].pop_size = island_info_.island_size / island_info_.subisland_num;
		}
	}
	if(island_info_.configure_EA == "constant")
	{
		for (int i = 0; i < island_info_.subisland_num; i++)
		{
			h_DE_info[i].CR = (real) DE_info_.CR;
			h_DE_info[i].F = (real) DE_info_.F;
			h_DE_info[i].strategy_ID = DE_info_.strategy_ID;
			h_DE_info[i].pop_size = island_info_.island_size / island_info_.subisland_num;
		}
	}
	cudaMemcpy(d_DE_info_, h_DE_info, sizeof(DEInfo) * island_info_.subisland_num, cudaMemcpyHostToDevice);

	delete[]h_DE_info;

	return 0;
}

int DE_CUDA::Reproduce_CPU(Population & candidate, Population & population)
{
    Individual best_individual = FindBestIndividual(population);
    int pop_size = population.size();
    vector<int> r = random_.Permutate(pop_size, 5);

    double F = DE_info_.F;
    double CR = DE_info_.CR;
    for (int i = 0; i < pop_size; i++)
    {
        Individual tmp_candidate;

        for (int j = 0; j < problem_info_.dim; j++)
        {
            double tmp_element = 0;
            switch (DE_info_.strategy_ID)
            {
                case 0:
                    tmp_element = population[i].elements[j] + F * (population[r[0]].elements[j] - population[r[1]].elements[j]);
                    break;
                case 1:
                    tmp_element = population[i].elements[j] + F * (population[r[0]].elements[j] - population[r[1]].elements[j]) + \
                    + F * (population[r[2]].elements[j] - population[r[3]].elements[j]);
                    break;
                case 2:
                    tmp_element = population[i].elements[j] + F * (best_individual.elements[j] - population[i].elements[j]) + \
                    + F * (population[r[0]].elements[j] - population[r[1]].elements[j]);
                    break;
                case 3:
                    tmp_element = population[i].elements[j] + F * (best_individual.elements[j] - population[i].elements[j]) + \
                    + F * (population[r[0]].elements[j] - population[r[1]].elements[j]) + F * (population[r[2]].elements[j] - population[r[3]].elements[j]);
                    break;
                case 4:
                    tmp_element = population[r[0]].elements[j] + F * (population[r[1]].elements[j] - population[r[2]].elements[j]);
                    break;
                case 5:
                    tmp_element = population[r[0]].elements[j] + F * (population[r[1]].elements[j] - population[r[2]].elements[j]) + \
                    + F * (population[r[3]].elements[j] - population[r[4]].elements[j]);
                    break;
                case 6:
                    tmp_element = best_individual.elements[j] + F * (population[r[0]].elements[j] - population[r[1]].elements[j]);
                    break;
                case 7:
                    tmp_element = best_individual.elements[j] + F * (population[r[0]].elements[j] - population[r[1]].elements[j]) + \
                    + F * (population[r[2]].elements[j] - population[r[3]].elements[j]);
                    break;
                case 8:
                    tmp_element = population[i].elements[j] + F * (population[r[0]].elements[j] - population[i].elements[j]) + \
                    + F * (population[r[1]].elements[j] - population[r[2]].elements[j]) + F * (population[r[3]].elements[j] - population[r[4]].elements[j]);
                    break;
                default:
                    break;
            }
            if (random_.RandRealUnif(0, 1) > CR && j != random_.RandIntUnif(0, problem_info_.dim - 1))
                tmp_element = population[i].elements[j];
            tmp_element = CheckBound(tmp_element, problem_info_.min_bound, problem_info_.max_bound);
            tmp_candidate.elements.push_back(tmp_element);
        }
        //tmp_candidate.fitness_value = cec2014_.EvaluateFitness(tmp_candidate.elements);
        candidate.push_back(tmp_candidate);
    }
    return 0;
}

int DE_CUDA::SelectSurvival_CPU(Population & population, Population & candidate)
{
	int pop_size = population.size();

    for (int i = 0; i < pop_size; i++)
    {
        if (candidate[i].fitness_value < population[i].fitness_value)
        {
            population[i].fitness_value = candidate[i].fitness_value;
            population[i].elements = candidate[i].elements;
        }
    }
    return 0;
}

int DE_CUDA::Run(Population & population)
{
#ifdef EA_CPU
    Population candidate;
	Reproduce_CPU(candidate, population);
    //TransferDataFromCPU(candidate);
    //cec2014_cuda_.EvaluateFitness(d_population_.fitness_value, d_population_.elements);
    //TransferDataToCPU(candidate);
	SelectSurvival_CPU(population, candidate);
#else
	API_DE_GenerateNewPopulation(d_candidate_, d_population_, d_best_individual_ID_, d_island_info_, d_DE_info_, d_rand_states_, problem_info_);
	cec2014_cuda_.EvaluateFitness(d_candidate_.fitness_value, d_candidate_.elements);
	API_DE_SelectSurvival(d_population_, d_candidate_, d_island_info_, problem_info_);
#endif

    return 0;
}
