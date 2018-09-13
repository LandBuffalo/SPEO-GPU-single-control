#include "config.h"
#include "island_EA.h"

#include <sstream>
#include <string>

char* getParam(char * needle, char* haystack[], int count) {
    int i = 0;
    for (i = 0; i < count; i++) {
        if (strcmp(needle, haystack[i]) == 0) {
            if (i < count - 1) {
                return haystack[i + 1];
            }
        }
    }
    return 0;
}

vector<string> &split(const string &str, char delim, vector<string> &elems, bool skip_empty = true)
{
    istringstream iss(str);
    for (string item; getline(iss, item, delim);)
        if (skip_empty && item.empty()) continue;
        else elems.push_back(item);
        return elems;
}

int SetParameters(vector<int> &total_functions, vector<int> &total_runs, IslandInfo &island_info, ProblemInfo &problem_info, NodeInfo &node_info, int argc, char** argv)
{
//---------------------------problem info----------------------------------------------------//
    problem_info.max_base_FEs = 10000000;
    problem_info.seed = 1;
    problem_info.max_bound = 100;
    problem_info.min_bound = -100;
    problem_info.dim = 100;
    problem_info.computing_time = 10;
    for (int i = 23; i <= 30; i++)
        total_functions.push_back(i);
    for (int i = 1; i <= 15; i++)
        total_runs.push_back(i);
    if (getParam("-dim", argv, argc))
        problem_info.dim = atoi(getParam("-dim", argv, argc));
    if (getParam("-total_functions", argv, argc))
    {
        total_functions.clear();
        string str = getParam("-total_functions", argv, argc);
        vector<string> functions;
        split(str, '-', functions);
        const char *tmp_function1 = functions[0].c_str();
        const char *tmp_function2 = functions[1].c_str();

        for (int i = atoi(tmp_function1); i <= atoi(tmp_function2); i++)
            total_functions.push_back(i);
    }
    if (getParam("-total_runs", argv, argc))
    {
        total_runs.clear();
        string str = getParam("-total_runs", argv, argc);
        vector<string> runs;
        split(str, '-', runs);
        const char *tmp_run1 = runs[0].c_str();
        const char *tmp_run2 = runs[1].c_str();

        for (int i = atoi(tmp_run1); i <= atoi(tmp_run2); i++)
            total_runs.push_back(i);

    }
    if (getParam("-computing_time", argv, argc))
        problem_info.computing_time = atoi(getParam("-computing_time", argv, argc));
    problem_info.computing_time = problem_info.computing_time / (total_runs.size() * total_functions.size() + 0.0);
    if (getParam("-max_base_FEs", argv, argc))
        problem_info.max_base_FEs = atoi(getParam("-max_base_FEs", argv, argc));
//---------------------------node info----------------------------------------------------//
    node_info.GPU_num = node_info.node_num;
    if (getParam("-GPU_num", argv, argc))
        node_info.GPU_num = atoi(getParam("-GPU_num", argv, argc));
    node_info.GPU_ID = node_info.node_ID;
		node_info.task_ID = 1;

//---------------------------island info----------------------------------------------------//
    island_info.interval = 10;
    island_info.island_num = node_info.GPU_num;
    island_info.island_size = 1024;
    island_info.migration_size = 32;
    island_info.buffer_capacity = 32;
    island_info.migration_rate = 0.5;
    island_info.migration_topology.assign("dynamicConnect");
    island_info.buffer_manage.assign("DP");
    island_info.configure_EA.assign("constant");
    island_info.regroup_option.assign("static_and_ordered");
    island_info.subisland_num = 1;//island_info.island_size / subisland_size;

  	if (getParam("-pop_size", argv, argc))
        island_info.island_size = atoi(getParam("-pop_size", argv, argc)) / island_info.island_num;
    if (getParam("-island_size", argv, argc))
        island_info.island_size = atoi(getParam("-island_size", argv, argc));
    if (getParam("-interval", argv, argc))
        island_info.interval = atoi(getParam("-interval", argv, argc));
    if (getParam("-migration_size", argv, argc))
        island_info.migration_size = atoi(getParam("-migration_size", argv, argc));
    if (getParam("-migration_topology", argv, argc))
        island_info.migration_topology = getParam("-migration_topology", argv, argc);
    if (getParam("-buffer_manage", argv, argc))
        island_info.buffer_manage = getParam("-buffer_manage", argv, argc);
    if (getParam("-regroup_option", argv, argc))
        island_info.regroup_option = getParam("-regroup_option", argv, argc);
    if (getParam("-configure_EA", argv, argc))
        island_info.configure_EA = getParam("-configure_EA", argv, argc);
    if (getParam("-buffer_capacity", argv, argc))
        island_info.buffer_capacity = atoi(getParam("-buffer_capacity", argv, argc));
    if (getParam("-migration_rate", argv, argc))
        island_info.migration_rate = atof(getParam("-migration_rate", argv, argc));
    if (getParam("-subisland_size", argv, argc))
    {
        int subisland_size = atoi(getParam("-subisland_size", argv, argc));
        island_info.subisland_num = island_info.island_size / subisland_size;
    }


    return 0;

}

int ConstructAndExecuteIslandModule(vector<int> &total_functions, vector<int> &total_runs, IslandInfo island_info, ProblemInfo problem_info, NodeInfo node_info)
{

    IslandEA island_EA(node_info);

    for (int i = 0; i < total_functions.size(); i++)
    {
        problem_info.function_ID = total_functions[i];
        if (node_info.node_ID == 1)
            printf("--------function_ID = %d---------------------\n", problem_info.function_ID);
        for (int j = 0; j < total_runs.size(); j++)
        {
            problem_info.run_ID = total_runs[j];

            problem_info.seed = (node_info.node_ID + problem_info.function_ID) * (problem_info.run_ID + problem_info.function_ID);
            srand(problem_info.seed);
            island_EA.Initilize(island_info, problem_info);
            island_EA.Execute();
            island_EA.Unitilize();
            MPI_Barrier(MPI_COMM_WORLD);

        }
        if (node_info.node_ID == 1)
            printf("-----------------------------------------------\n");
    }
    return 0;
}


int main(int argc, char* argv[])
{
    int current_node_id = 0;
    int total_node_num = 0;
    double start_time, finish_time;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &current_node_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_node_num);

    IslandInfo island_info;
    ProblemInfo problem_info;
    NodeInfo node_info;

    node_info.node_num = total_node_num;
    node_info.node_ID = current_node_id;
    start_time = MPI_Wtime();
    srand(time(NULL) * current_node_id);
    vector<int> total_functions, total_runs;

    SetParameters(total_functions, total_runs, island_info, problem_info, node_info, argc, argv);

    ConstructAndExecuteIslandModule(total_functions, total_runs, island_info, problem_info, node_info);

    finish_time = MPI_Wtime();
    MPI_Finalize();

    printf("Total Elapsed time: %f seconds\n", finish_time - start_time);
    return 0;
}
