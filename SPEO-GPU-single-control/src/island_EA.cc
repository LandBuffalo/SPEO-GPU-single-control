#include "island_EA.h"
IslandEA::IslandEA(const NodeInfo node_info):migrate_(node_info)
{
    node_info_ = node_info;
    CheckAndCreatRecordFile();
}

IslandEA::~IslandEA()
{

}

int IslandEA::CheckAndCreatRecordFile()
{
    file_name_ = "./Results/SPEO_GPU.csv";
    ifstream exist_file;
    exist_file.open(file_name_.c_str());
    ofstream file;

    if(!exist_file)
    {
        file.open(file_name_.c_str());
        file<< "function_ID,run_ID,dim,best_fitness,time_period,communication_percentage,total_FEs,node_num,GPU_num,island_num,total_pop_size,subisland_size,interval,migration_size,migration_rate,buffer_capacity,migration_topology,buffer_manage,regroup_option,configure_EA,EA_parameters"<<endl;
        file.close();
    }
    else
        exist_file.close();

    return 0;

}

int IslandEA::Initilize(IslandInfo island_info, ProblemInfo problem_info)
{
    problem_info_ = problem_info;
    island_info_ = island_info;
    EA_CUDA_ = new DE_CUDA(node_info_);
    EA_CUDA_->Initilize(problem_info_, island_info_);
	EA_CUDA_->InitilizePopulation(sub_population_);
    migrate_.Initilize(island_info_, problem_info_);

    return 0;
}

int IslandEA::Unitilize()
{
    sub_population_.clear();
    EA_CUDA_->Unitilize();
    migrate_.Unitilize();

    return 0;
}

int IslandEA::Finish()
{
    if(node_info_.GPU_ID != 0)
    {
        SendResultToIsland0();
    }
    else
    {
        vector<DisplayUnit> total_display_unit;
        RecvResultFromOtherIsland(total_display_unit);
        MergeResults(total_display_unit);
        PrintResult();
    }
    return 0;
}


int IslandEA::RunEA()
{
    EA_CUDA_->Run(sub_population_);

    return 0;
}

int IslandEA::Execute()
{
    int generation = 0;
    int current_FEs = island_info_.island_size;
    double start_time = MPI_Wtime();
    double current_time = 0;
    real communication_time = 0;

    long int total_FEs = problem_info_.max_base_FEs * problem_info_.dim / island_info_.island_num;
#ifndef COMPUTING_TIME
    while(current_FEs < total_FEs)
#else
    while(MPI_Wtime() - start_time < problem_info_.computing_time)
#endif
    {
        RunEA();
        if (generation % island_info_.interval == 0)
        {
            double tmp_time = MPI_Wtime();
            migrate_.MigrateOut(EA_CUDA_, sub_population_);
            communication_time += (real) (MPI_Wtime() - tmp_time);
        }
        double tmp_time = MPI_Wtime();
        migrate_.MigrateIn(EA_CUDA_, sub_population_);
        communication_time += (real) (MPI_Wtime() - tmp_time);

        generation++;
        current_FEs += island_info_.island_size;
    }
    RecordDisplayUnit((real) (MPI_Wtime() - start_time), communication_time);

    Finish();

    return 0;

}

int IslandEA::RecordDisplayUnit(real current_time, real communication_time)
{
    display_unit_.time = current_time;
    display_unit_.communication_percentage = communication_time / display_unit_.time;
    display_unit_.fitness_value = EA_CUDA_->FindBestIndividual(sub_population_).fitness_value;
    return 0;
}

int IslandEA::MergeResults(vector<DisplayUnit> &total_display_unit)
{
    display_unit_.fitness_value = total_display_unit[0].fitness_value;
    for(int i = 1; i < node_info_.GPU_num; i++)
    {
        if(display_unit_.fitness_value > total_display_unit[i].fitness_value)
            display_unit_.fitness_value = total_display_unit[i].fitness_value;
    }

    display_unit_.time = total_display_unit[0].time;
    display_unit_.communication_percentage = total_display_unit[0].communication_percentage;
    for(int i = 1; i < node_info_.GPU_num; i++)
    {
        display_unit_.time += total_display_unit[i].time;
        display_unit_.communication_percentage += total_display_unit[i].communication_percentage;
    }
    display_unit_.time = display_unit_.time / (node_info_.GPU_num + 0.0);
    display_unit_.communication_percentage = display_unit_.communication_percentage / (node_info_.GPU_num + 0.0);

    printf("Run: %d, Time: %.3lf s, Best fitness: %.9lf\n", problem_info_.run_ID, display_unit_.time, display_unit_.fitness_value);

}

int IslandEA::PrintResult()
{
    ofstream file;
    file.open(file_name_.c_str(), ios::app);

    int function_ID = problem_info_.function_ID;
    int run_ID = problem_info_.run_ID;
    int dim = problem_info_.dim;

    long int total_FEs = problem_info_.max_base_FEs * problem_info_.dim;
    int total_pop_size = island_info_.island_num * island_info_.island_size;

    string EA_parameters = EA_CUDA_->GetParameters();
    real best_fitness = display_unit_.fitness_value;
    real time_period = display_unit_.time;
    real communication_percentage = display_unit_.communication_percentage;
    int subisland_size = -1;
    if(island_info_.regroup_option != "dynamic_and_random" && island_info_.regroup_option != "dynamic_and_ordered")
        subisland_size = island_info_.island_size / island_info_.subisland_num;

    file<<function_ID<<','<<run_ID<<','<<dim<<','<<best_fitness<<','<<time_period<<','<<communication_percentage<<','<<total_FEs<<','<<node_info_.node_num\
    <<','<<node_info_.GPU_num<<','<<island_info_.island_num<<','<<total_pop_size<<','<<subisland_size<<','<<island_info_.interval<<','<<island_info_.migration_size\
    <<','<<island_info_.migration_rate<<','<<island_info_.buffer_capacity<<','<<island_info_.migration_topology<<','<<island_info_.buffer_manage<<','<<island_info_.regroup_option<<','<<island_info_.configure_EA<<','<<EA_parameters<<endl;
    file.close();
    return 0;
}

int IslandEA::SendResultToIsland0()
{
    real *msg = new real[3];
    msg[0] = display_unit_.time;
    msg[1] = display_unit_.communication_percentage;
    msg[2] = display_unit_.fitness_value;

    int tag = problem_info_.function_ID * 1000 +  10 * problem_info_.run_ID + FLAG_DISPLAY_UNIT;
#ifdef GPU_DOUBLE_PRECISION
    MPI_Send(msg, 3, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
#endif
#ifdef GPU_SINGLE_PRECISION
    MPI_Send(msg, 3, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
#endif
    delete [] msg;

    return 0;
}

int IslandEA::RecvResultFromOtherIsland(vector<DisplayUnit> &total_display_unit)
{
    MPI_Status mpi_status;
    DisplayUnit tmp_display_unit;
    tmp_display_unit.time = display_unit_.time;
    tmp_display_unit.communication_percentage = display_unit_.communication_percentage;
    tmp_display_unit.fitness_value = display_unit_.fitness_value;
    total_display_unit.push_back(tmp_display_unit);
    real *msg = new real[3];
    for(int i = 1; i < node_info_.GPU_num; i++)
    {
        int tag = problem_info_.function_ID * 1000 +  10 * problem_info_.run_ID + FLAG_DISPLAY_UNIT;
#ifdef GPU_DOUBLE_PRECISION
        MPI_Recv(msg, 3, MPI_DOUBLE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &mpi_status);
#endif
#ifdef GPU_SINGLE_PRECISION
        MPI_Recv(msg, 3, MPI_FLOAT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &mpi_status);
#endif
        DisplayUnit tmp_display_unit;
        tmp_display_unit.time = msg[0];
        tmp_display_unit.communication_percentage = msg[1];
        tmp_display_unit.fitness_value = msg[2];
        total_display_unit.push_back(tmp_display_unit);
    }
    delete []msg;
    return 0;
}
