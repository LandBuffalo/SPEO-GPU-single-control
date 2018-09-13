#ifndef __CUDA_EA_LIBRARY_H__
#define __CUDA_EA_LIBRARY_H__
#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>
#include "config.h"
#include "CEC2014_CUDA.h"
#include "CEC2014.h"

#include "random.h"
#include <sstream>

#define MAX_THREAD 512
#define WARP_SIZE 32
#define MAX_POOL_SIZE 64
#define TILE_WIDTH 16


struct GPU_Population
{
    real *elements;
    real *fitness_value;
};
struct GPU_IslandInfo
{
    int island_size;
  int subisland_num;
    int *permutated_index;
};

struct DEInfo
{
    real CR;
    real F;
    int strategy_ID;
    int pop_size;
};

extern "C"
int API_Initilize(GPU_Population d_population, curandState *d_rand_states, GPU_IslandInfo GPU_island_info, ProblemInfo problem_info);
extern "C"
int API_CheckCUDAError();

class EA_CUDA
{
protected:
    ProblemInfo             problem_info_;
    IslandInfo              island_info_;
    NodeInfo                node_info_;
    CEC2014_CUDA            cec2014_cuda_;
    CEC2014                 cec2014_;

    Random                  random_;
    GPU_Population          d_population_;
    GPU_Population          d_candidate_;
    GPU_Population          h_population_;
    GPU_IslandInfo          d_island_info_;

    curandState *           d_rand_states_;
    int                     CalNextPow2Dim();
    real                  CheckBound(real to_check_elements, real min_bound, real max_bound);

public:
                            EA_CUDA();
                            ~EA_CUDA();

    int                     InitilizePopulation(Population & population);
    virtual int             Initilize(ProblemInfo problem_info, IslandInfo island_info);
    virtual int             Unitilize();

    int                     TransferDataToCPU(Population &population);
    int                     TransferDataFromCPU(Population &population);

    virtual Individual      FindBestIndividual(Population & population);
    int                     RegroupIslands(vector<int> &permutate_index, IslandInfo island_info);
    Population              FindBestIndividualInIslands(Population & population);
    virtual int             Run(Population & population) = 0;
    virtual string          GetParameters() = 0;
    virtual int             ConfigureEA()=0;

};

class DE_CUDA : public EA_CUDA
{
private:
    DEInfo                  DE_info_;
    DEInfo *                d_DE_info_;

    int *                   d_best_individual_ID_;
    int                     Reproduce_CPU(Population & candidate, Population & population);
    int                     SelectSurvival_CPU(Population & population, Population & candidate);
public:
                            DE_CUDA(NodeInfo node_info);
                            ~DE_CUDA();
    virtual int             Initilize(ProblemInfo problem_info, IslandInfo island_info);
    virtual int             Unitilize();
    virtual int             Run(Population & population);
    virtual string          GetParameters();
    virtual int             ConfigureEA();

};


#endif
