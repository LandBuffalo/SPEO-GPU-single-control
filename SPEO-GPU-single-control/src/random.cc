#include "random.h"


Random::Random()
{
}


Random::~Random()
{

}

int Random::RandIntUnif(int min_value, int max_value)
{
	if (min_value != max_value)
		return min_value + rand() % (max_value - min_value + 1);
	else
		return min_value;
}
real Random::RandRealUnif(real min_value, real max_value)
{

	if (min_value != max_value)
		return min_value + rand() / (RAND_MAX + 0.0) * (max_value - min_value + 0.0);
	else
		return min_value;
}
vector<int> Random::Permutate(int arrary_length, int permutation_length)
{
	if(permutation_length > arrary_length)
	{
		printf("permutation_length=%d\arrary_length=%d\t", permutation_length, arrary_length);
		permutation_length = arrary_length;
	}
	vector<int> permutate_index;
	vector<int> global_perm(arrary_length);
	for (int local_ind_individual = 0; local_ind_individual < arrary_length; local_ind_individual++)
		global_perm[local_ind_individual] = local_ind_individual;

	int tmp = 0;
	int i = arrary_length;

	while (i > arrary_length - permutation_length)     //pm_depth is the number of random indices wanted (must be <= NP)
	{
		tmp = RandIntUnif(0, (i - 1));
		permutate_index.push_back(global_perm[tmp]);
		global_perm[tmp] = global_perm[i - 1];
		i--;
	}

	return permutate_index;
}

vector<int>	Random::Permutate(int arrary_length, int permutation_length, vector<int> &avoid_index)
{
	if(permutation_length > arrary_length)
	{
		printf("permutation_length=%d\arrary_length=%d\t", permutation_length, arrary_length);
		permutation_length = arrary_length;
	}
	vector<int> permutate_index;
	vector<int> global_perm(arrary_length);
	for (int local_ind_individual = 0; local_ind_individual < arrary_length; local_ind_individual++)
		global_perm[local_ind_individual] = local_ind_individual;

	int tmp = 0;
	int i = arrary_length;

	while (i > arrary_length - permutation_length)     //pm_depth is the number of random indices wanted (must be <= NP)
	{
		tmp = RandIntUnif(0, (i - 1));
        for (int j = 0; j < avoid_index.size(); j++)
        {
            if (global_perm[tmp] == avoid_index[i])
            {
                j = 0;
                tmp = RandIntUnif(0, (i - 1));
            }

        }

		permutate_index.push_back(global_perm[tmp]);
		global_perm[tmp] = global_perm[i - 1];
		i--;
	}

	return permutate_index;
}

int Random::Permutate(vector<int> & permutate_index, int arrary_length, int permutation_length)
{
	if(permutation_length > arrary_length)
	{
		printf("permutation_length=%d\arrary_length=%d\t", permutation_length, arrary_length);
		permutation_length = arrary_length;
	}
	vector<int> global_perm(arrary_length);
	for (int local_ind_individual = 0; local_ind_individual < arrary_length; local_ind_individual++)
		global_perm[local_ind_individual] = local_ind_individual;

	int tmp = 0;
	int i = arrary_length;

	while (i > arrary_length - permutation_length)     //pm_depth is the number of random indices wanted (must be <= NP)
	{
		tmp = RandIntUnif(0, (i - 1));
		permutate_index.push_back(global_perm[tmp]);
		global_perm[tmp] = global_perm[i - 1];
		i--;
	}
	return 0;

}

int Random::Permutate(int * permutate_index, int arrary_length, int permutation_length)
{
	if(permutation_length > arrary_length)
	{
		printf("permutation_length=%d\arrary_length=%d\t", permutation_length, arrary_length);
		permutation_length = arrary_length;
	}
	vector<int> global_perm(arrary_length);
	for (int local_ind_individual = 0; local_ind_individual < arrary_length; local_ind_individual++)
		global_perm[local_ind_individual] = local_ind_individual;

	int tmp = 0;
	int i = arrary_length;
	int count = 0;
	while (i > arrary_length - permutation_length)     //pm_depth is the number of random indices wanted (must be <= NP)
	{
		tmp = RandIntUnif(0, (i - 1));
		permutate_index[count] = global_perm[tmp];
		global_perm[tmp] = global_perm[i - 1];
		i--;
		count++;
	}
	return 0;
}
