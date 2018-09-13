#ifndef __RANDOM_H__
#define __RANDOM_H__
#pragma once
#include "config.h"
class Random
{
public:
	Random();
	~Random();
	int			         Permutate(vector<int> & requested_island_ID, int arrary_length, int permutation_length);
	vector<int>	           Permutate(int arrary_length, int permutation_length);
	vector<int>	           Permutate(int arrary_length, int permutation_length, vector<int> &avoid_index);

	int 			Permutate(int * permutate_index, int arrary_length, int permutation_length);
	int			     RandIntUnif(int min_value, int max_value);
	real 			RandRealUnif(real min_value, real max_value);


};

#endif
