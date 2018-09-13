#ifndef __ISLANDEA_H__
#define __ISLANDEA_H__
#pragma once
#include "config.h"
#include "random.h"
#include "EA_CUDA.h"
#include "migrate.h"

class IslandEA
{
private:
	EA_CUDA	*				EA_CUDA_;
	Random                  random_;
	NodeInfo				node_info_;
	Migrate 				migrate_;

	ProblemInfo				problem_info_;
	IslandInfo				island_info_;

	Individual 				best_individuals_;
    Population 				sub_population_;

	DisplayUnit		 		display_unit_;
    string 					file_name_;

	int						RunEA();
    int                     RecordDisplayUnit(real current_time, real communication_time);
	int 					PrintResult();
	int 					RecvResultFromOtherIsland(vector<DisplayUnit> &total_display_unit);
	int 					MergeResults(vector<DisplayUnit> &total_display_unit);
	int 					CheckAndCreatRecordFile();
	int 					SendResultToIsland0();
    int                    	Finish();
    int         			RegroupIsland();
public:
							IslandEA(const NodeInfo node_info);
							~IslandEA();
	int 					Initilize(IslandInfo island_info, ProblemInfo problem_info);
	int 					Unitilize();
	int						Execute();
};

#endif
