#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <cmath>
#include <time.h>
#include <numeric>
#include <string>
#include <iostream>
#include <iomanip>

using namespace std;

#define TOTAL_RECORD_NUM		20
#define EMIGRATIONS_ISLAND     0
#define EMIGRATIONS_EA     1
#define FLAG_FINISH       		2
#define FLAG_DISPLAY_UNIT       3


#define MIN_SUBISLAND_SIZE      4

//#define DISPLAY
//#define DIVERSITY
//#define EA_CPU
//#define COMPUTING_TIME

#define GPU_DOUBLE_PRECISION
#ifdef GPU_DOUBLE_PRECISION
  typedef double real;
#endif
#ifdef GPU_SINGLE_PRECISION
  typedef float real;
#endif

struct Individual
{
	vector<real> elements;
	real fitness_value;
};

typedef vector<Individual> Population;


struct ProblemInfo
{
	int dim;
	int function_ID;
	int run_ID;
	int max_base_FEs;
	int seed;
	int running_time;
	int computing_time;
	real max_bound;
	real min_bound;
};

struct NodeInfo
{
    int task_ID;
	int node_ID;
	int node_num;
	int GPU_num;
	int GPU_ID;
};

struct IslandInfo
{
	int island_size;
	int island_num;
	int interval;
	int migration_size;
    int buffer_capacity;
	real migration_rate;
  	int subisland_num;
    string configure_EA;
    string regroup_option;
    string migration_topology;
	string buffer_manage;
};

struct DisplayUnit
{
	real time;
	real communication_percentage;
	real fitness_value;
};

//#define DEBUG


#endif
