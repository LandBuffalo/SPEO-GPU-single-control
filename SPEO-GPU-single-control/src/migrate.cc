#include "migrate.h"

Migrate::Migrate(NodeInfo node_info)
{
    node_info_ = node_info;
}

Migrate::~Migrate()
{

}

int Migrate::Initilize(IslandInfo island_info, ProblemInfo problem_info)
{
    island_info_ = island_info;
    problem_info_ = problem_info;
    int message_length = island_info_.migration_size * (problem_info_.dim + 1);
    send_msg_to_other_EA_ = new real[message_length];
    buffer_manage_ = new OnlineCluster(island_info.buffer_capacity);
    for(int i = 0; i < island_info_.island_size; i++)
        regroup_permutated_index_.push_back(i);
    success_sent_flag_ = 1;
    return 0;
}
int Migrate::Unitilize()
{
    delete []send_msg_to_other_EA_;
    delete buffer_manage_;
    regroup_permutated_index_.clear();

    return 0;
}

int Migrate::UpdateDestination()
{
    if(destinations_.size() == 0)
    {
        for (int i = 0; i < node_info_.GPU_num; i++)
        {
            if(random_.RandRealUnif(0,1) < island_info_.migration_rate && i != node_info_.GPU_ID)
                destinations_.push_back(i);
        }
    }
}

int Migrate::UpdatePopulation(EA_CUDA *EA_CUDA, Population & population)
{
    Population emigration_import;
    buffer_manage_->SelectFromBuffer(emigration_import, island_info_.migration_size);
	vector<int> permutate_index;
    if (emigration_import.size() != 0)
    {
        permutate_index = random_.Permutate(population.size(), emigration_import.size());
        EA_CUDA->TransferDataToCPU(population);
        for (int i = 0; i < emigration_import.size(); i++)
            population[permutate_index[i]] = emigration_import[i];
        EA_CUDA->TransferDataFromCPU(population);
    }
    RegroupIslands(EA_CUDA, population);

    return 0;
}

int Migrate::MigrateOut(EA_CUDA *EA_CUDA, Population &population)
{
    UpdateDestination();
	UpdatePopulation(EA_CUDA, population);
    return 0;
}

int Migrate::CheckAndRecvEmigrations()
{
 	MPI_Status mpi_status;
    int flag = 0;
    int tag = 1000 * problem_info_.function_ID + 10 * problem_info_.run_ID + EMIGRATIONS_ISLAND;
    MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &mpi_status);
    if(flag == 1)
    {
        int count = 0;
#ifdef GPU_DOUBLE_PRECISION
        MPI_Get_count(&mpi_status, MPI_DOUBLE, &count);
#endif
#ifdef GPU_SINGLE_PRECISION
        MPI_Get_count(&mpi_status, MPI_FLOATL, &count);
#endif
        int message_length = count;
        real * msg_recv = new real[message_length];
        int source = mpi_status.MPI_SOURCE;
#ifdef GPU_DOUBLE_PRECISION
        MPI_Recv(msg_recv, message_length, MPI_DOUBLE, source, mpi_status.MPI_TAG, MPI_COMM_WORLD, &mpi_status);
#endif
#ifdef GPU_SINGLE_PRECISION
        MPI_Recv(msg_recv, message_length, MPI_FLOATL, source, mpi_status.MPI_TAG, MPI_COMM_WORLD, &mpi_status);
#endif
        Population emigration_import;
        DeserialMsgToIndividual(emigration_import, msg_recv, count / (problem_info_.dim + 1));

        buffer_manage_->UpdateBuffer(emigration_import);
        delete [] msg_recv;
    }

    return 0;
}

int Migrate::SendEmigrations(EA_CUDA *EA_CUDA, Population &population)
{
    MPI_Status mpi_status;

    if(success_sent_flag_ == 0)
    {
        MPI_Test(&mpi_request_, &success_sent_flag_, &mpi_status);
    }
    if (success_sent_flag_ == 1 && destinations_.size() > 0)
    {
        Population emigration_export;
        EA_CUDA->TransferDataToCPU(population);

        vector<int> tmp_ID = random_.Permutate(population.size(), island_info_.migration_size);
        for(int i = 0; i < island_info_.migration_size; i++)
            emigration_export.push_back(population[tmp_ID[i]]);

        int message_length = island_info_.migration_size * (problem_info_.dim + 1);
        SerialIndividualToMsg(send_msg_to_other_EA_, emigration_export);
    	int tag = 1000 * problem_info_.function_ID + 10 * problem_info_.run_ID + EMIGRATIONS_ISLAND;
#ifdef GPU_DOUBLE_PRECISION
        MPI_Isend(send_msg_to_other_EA_, message_length, MPI_DOUBLE, destinations_[0], tag, MPI_COMM_WORLD, &mpi_request_);
#endif
#ifdef GPU_SINGLE_PRECISION
        MPI_Isend(send_msg_to_other_EA_, message_length, MPI_FLOAT, destinations_[0], tag, MPI_COMM_WORLD, &mpi_request_);
#endif
        destinations_.erase(destinations_.begin());
        success_sent_flag_ = 0;
    }

    return 0;
}

int Migrate::MigrateIn(EA_CUDA *EA_CUDA, Population &population)
{
	SendEmigrations(EA_CUDA, population);
	CheckAndRecvEmigrations();

    return 0;
}


int Migrate::RegroupIslands(EA_CUDA *EA_CUDA, Population &population)
{
    if(island_info_.regroup_option == "dynamic_and_random")
    {
        int max_index = 0, pop_size = population.size() / MIN_SUBISLAND_SIZE;
        while (pop_size > 1)
        {
            pop_size = pop_size / 2;
            max_index++;
        }
        int subisland_size = MIN_SUBISLAND_SIZE;
        subisland_size = subisland_size << random_.RandIntUnif(0, max_index);
        island_info_.subisland_num = population.size() / subisland_size;
        regroup_permutated_index_.clear();
        regroup_permutated_index_ = random_.Permutate(population.size(), population.size());
        EA_CUDA->RegroupIslands(regroup_permutated_index_, island_info_);
        EA_CUDA->ConfigureEA();
    }
   if(island_info_.regroup_option == "dynamic_and_ordered")
    {
        int max_index = 0, pop_size = population.size() / MIN_SUBISLAND_SIZE;
        while (pop_size > 1)
        {
            pop_size = pop_size / 2;
            max_index++;
        }
        int subisland_size = MIN_SUBISLAND_SIZE;
        subisland_size = subisland_size << random_.RandIntUnif(0, max_index);
        island_info_.subisland_num = population.size() / subisland_size;
        EA_CUDA->RegroupIslands(regroup_permutated_index_, island_info_);
        EA_CUDA->ConfigureEA();
    }
    if(island_info_.regroup_option == "static_and_random")
    {
        regroup_permutated_index_.clear();
        regroup_permutated_index_ = random_.Permutate(population.size(), population.size());
        EA_CUDA->RegroupIslands(regroup_permutated_index_, island_info_);
    }
    return 0;
}

vector<int> Migrate::FindBestIndividualInIsland(Population &population)
{
    int subisland_size = island_info_.island_size / island_info_.subisland_num;
    vector<int> best_individual_index;
    for(int i = 0; i < island_info_.subisland_num; i++)
    {
        int tmp_best_individual_index = regroup_permutated_index_[i * subisland_size];
        real best_fitness_value = population[tmp_best_individual_index].fitness_value;
        for (int j = 1; j < subisland_size; j++)
        {
            int individual_index = regroup_permutated_index_[i * subisland_size + j];
            if(best_fitness_value > population[individual_index].fitness_value)
            {
                tmp_best_individual_index = individual_index;
                best_fitness_value = population[individual_index].fitness_value;
            }
        }
        best_individual_index.push_back(tmp_best_individual_index);
    }
    return best_individual_index;
}


int Migrate::DeserialMsgToIndividual(vector<Individual> &individual, real *msg, int length)
{
    int count = 0;

    for (int i = 0; i < length; i++)
    {
        Individual local_individual;
        for(int j = 0; j < problem_info_.dim; j++)
        {
            local_individual.elements.push_back(msg[count]);
            count++;
        }
        local_individual.fitness_value = msg[count];
        count++;
        individual.push_back(local_individual);
    }
    return 0;
}


int Migrate::SerialIndividualToMsg(real *msg, vector<Individual> &individual)
{
    int count = 0;
    for (int i = 0; i < individual.size(); i++)
    {
        for (int j = 0; j < problem_info_.dim; j++)
        {
            msg[count] = individual[i].elements[j];
            count++;
        }
        msg[count] = individual[i].fitness_value;
        count++;
    }
    return 0;
}
