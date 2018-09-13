#ifndef __MIGRATE_HH__
#define __MIGRATE_HH__
#include <mpi.h>
#include "random.h"
#include "config.h"
#include "buffer_manage.h"
#include "EA_CUDA.h"

class Migrate
{
private:
    Random                  random_;
    IslandInfo              island_info_;
    ProblemInfo             problem_info_;
    NodeInfo                node_info_;
    
    MPI_Request             mpi_request_;
    int                     success_sent_flag_;
    real *                  send_msg_to_other_EA_;
    BufferManage *          buffer_manage_;
    vector<int>             destinations_;
    vector<int>             regroup_permutated_index_;

    int                     RegroupIslands(EA_CUDA *EA_CUDA, Population &population);
    vector<int>             FindBestIndividualInIsland(Population &population);
    int                     SerialIndividualToMsg(real *msg_of_node_EA, vector<Individual> &individual);
    int                     DeserialMsgToIndividual(vector<Individual> &individual, real *msg_of_node_EA, int length);
    
    int                     UpdateDestination();
    int                     UpdatePopulation(EA_CUDA *EA_CUDA, Population & population);
    int                     CheckAndRecvEmigrations();
    int                     SendEmigrations(EA_CUDA *EA_CUDA, Population &population);
public:
                            Migrate(const NodeInfo node_info);
                            ~Migrate();
    int                     Initilize(IslandInfo island_info, ProblemInfo problem_info);
    int                     Unitilize();
    int                     MigrateOut(EA_CUDA *EA_CUDA, Population &population);
    int                     MigrateIn(EA_CUDA *EA_CUDA, Population &population);
};
#endif
