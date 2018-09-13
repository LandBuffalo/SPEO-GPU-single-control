#include "buffer_manage.h"


BufferManage::BufferManage(int buffer_capacity) : buffer_capacity_(buffer_capacity)
{
}

BufferManage::BufferManage()
{

}
BufferManage::~BufferManage()
{

}

real BufferManage::CalDiversity()
{
    real sum = 0;
    if(recv_buffer_.size() > 0)
    {
        for(int i = 0; i < recv_buffer_.size(); i++)
            sum += recv_buffer_[i].fitness_value;
        real mean = sum / (recv_buffer_.size() + 0.0);
        sum = 0;
        for(int i = 0; i < recv_buffer_.size(); i++)
            sum += (recv_buffer_[i].fitness_value - mean) * (recv_buffer_[i].fitness_value - mean);
        return sqrt(sum / (recv_buffer_.size() + 0.0));
    }
    else
    {
        return 0;
    }
}

OnlineCluster::OnlineCluster(int buffer_capacity):BufferManage(buffer_capacity)
{

}

OnlineCluster::~OnlineCluster()
{

}

real OnlineCluster::CalDistance(Individual &individual1, Individual &individual2)
{
    real distance_sum = 0;
    int dim = individual1.elements.size();
    for(int i = 0; i < dim; i++)
        distance_sum += (individual1.elements[i] - individual2.elements[i]) * (individual1.elements[i] - individual2.elements[i]);
    //distance_sum = abs(individual1.fitness_value - individual2.fitness_value);
    distance_sum = sqrt(distance_sum);

    return distance_sum;
}

int OnlineCluster::CreatCluster(Individual &individual)
{
    recv_buffer_.push_back(individual);
    cluster_centre_.push_back(individual);
    cluster_member_num_.push_back(1);
    flag_updated_.push_back(1);
    return 0;
}

int OnlineCluster::ReplaceMemberToCluster(int belong_cluster_ID, Individual &individual)
{
    int dim = individual.elements.size();

    for(int i = 0; i < dim; i++)
        cluster_centre_[belong_cluster_ID].elements[i] = (cluster_member_num_[belong_cluster_ID] * cluster_centre_[belong_cluster_ID].elements[i] + individual.elements[i]) / (cluster_member_num_[belong_cluster_ID] + 1.0);

    cluster_member_num_[belong_cluster_ID]++;
   // if(recv_buffer_[belong_cluster_ID].fitness_value > individual.fitness_value)
    //{
      recv_buffer_[belong_cluster_ID] = individual;
      flag_updated_[belong_cluster_ID] = 1;
   // }

    return 0;
}

int OnlineCluster::SelectFromBuffer(Population &emigration_export, int emigration_num)
{//for(int i = 0; i < recv_buffer_.size() && emigration_export.size() < emigration_num; i++)
    for(int i = 0; i < recv_buffer_.size(); i++)        
    {
        if(flag_updated_[i] == 1)
        {
            emigration_export.push_back(recv_buffer_[i]);
            flag_updated_[i] = 0;

        }
    }
    return 0;
}

int OnlineCluster::UpdateBuffer(Population &emigration_import)
{
    for(int i = 0; i < emigration_import.size(); i++)
    {
        if(recv_buffer_.size() < buffer_capacity_)
            CreatCluster(emigration_import[i]);
        else
        {
            int belong_cluster_ID = FindNearestIndividual(emigration_import[i], cluster_centre_);
           ReplaceMemberToCluster(belong_cluster_ID, emigration_import[i]);
        }
    }
    return 0;
}


int OnlineCluster::FindNearestIndividual(Individual &individual, Population &population)
{
    vector<real> distance_sum;
    for(int i = 0; i < population.size(); i++)
        distance_sum.push_back(CalDistance(individual, population[i]));

    int best_index = 0;
    real best_value = distance_sum[0];
    for(int i = 1; i < distance_sum.size(); i++)
    {
        if(best_value > distance_sum[i])
        {
            best_index = i;
            best_value = distance_sum[i];
        }
    }
    return best_index;
}

real OnlineCluster::CalDiversity()
{
    return BufferManage::CalDiversity();
/*
    real sum = 0;
    int total_candidates = 0;
    real mean = 0;
    if(recv_buffer_.size() > 0)
    {
        for(int i = 0; i < recv_buffer_.size(); i++)
        {
            if (flag_updated_[i] == 1)
            {
                total_candidates++;
                sum += recv_buffer_[i].fitness_value;
            }
        }
        if (total_candidates > 0)
            mean = sum / (total_candidates + 0.0);
        else
            return 0;
        sum = 0;
        for(int i = 0; i < recv_buffer_.size(); i++)
        {
            if (flag_updated_[i] == 1)
                sum += (recv_buffer_[i].fitness_value - mean) * (recv_buffer_[i].fitness_value - mean);
        }
        return sqrt(sum / (total_candidates + 0.0));
    }
    else
    {
        return 0;
    }
    */
}

BestSelect::BestSelect(int buffer_capacity):BufferManage(buffer_capacity)
{

}

BestSelect::~BestSelect()
{

}

int BestSelect::UpdateBuffer(Population &emigration_import)
{
    for(int i = 0; i < emigration_import.size(); i++)
    {
        recv_buffer_.push_back(emigration_import[i]);
        if(recv_buffer_.size() > buffer_capacity_)
            recv_buffer_.erase(recv_buffer_.begin());
    }


    return 0;
}

int BestSelect::FindBestBufferMember()
{
    int best_member_ID = 0;
    real best_member_fitness_value = recv_buffer_[0].fitness_value;
    for(int i = 1; i < recv_buffer_.size(); i++)
    {
        if(best_member_fitness_value < recv_buffer_[i].fitness_value)
        {
            best_member_ID = i;
            best_member_fitness_value = recv_buffer_[i].fitness_value;

        }
    }
    return best_member_ID;
}

int BestSelect::SelectFromBuffer(Population &emigration_export, int emigration_num)
{
    for(int i = 0; i < emigration_num && recv_buffer_.size() > 0; i++)
    {
        int temp_index = FindBestBufferMember();
        emigration_export.push_back(recv_buffer_[temp_index]);
        recv_buffer_.erase(recv_buffer_.begin() + temp_index);
    }
    return 0;
}


RandomSelect::RandomSelect(int buffer_capacity):BufferManage(buffer_capacity)
{

}

RandomSelect::~RandomSelect()
{

}

int RandomSelect::UpdateBuffer(Population &emigration_import)
{
    for(int i = 0; i < emigration_import.size(); i++)
    {
        recv_buffer_.push_back(emigration_import[i]);
        if(recv_buffer_.size() > buffer_capacity_)
            recv_buffer_.erase(recv_buffer_.begin());
    }

    return 0;
}
int RandomSelect::SelectFromBuffer(Population &emigration_export, int emigration_num)
{
    for(int i = 0; i < emigration_num && recv_buffer_.size() > 0; i++)
    {
        int temp_index = random_.RandIntUnif(0, recv_buffer_.size() - 1);
        emigration_export.push_back(recv_buffer_[temp_index]);
        recv_buffer_.erase(recv_buffer_.begin() + temp_index);
    }
    return 0;
}

FIFO::FIFO(int buffer_capacity):BufferManage(buffer_capacity)
{

}

FIFO::~FIFO()
{

}

int FIFO::UpdateBuffer(Population &emigration_import)
{
    for(int i = 0; i < emigration_import.size(); i++)
    {
        recv_buffer_.push_back(emigration_import[i]);
        if(recv_buffer_.size() > buffer_capacity_)
            recv_buffer_.erase(recv_buffer_.begin());
    }

    return 0;
}

int FIFO::SelectFromBuffer(Population &emigration_export, int emigration_num)
{
    for(int i = 0; i < emigration_num && recv_buffer_.size() > 0; i++)
    {
        emigration_export.push_back(recv_buffer_[0]);
        recv_buffer_.erase(recv_buffer_.begin());
    }
    return 0;
}

LIFO::LIFO(int buffer_capacity):BufferManage(buffer_capacity)
{

}

LIFO::~LIFO()
{

}

int LIFO::UpdateBuffer(Population &emigration_import)
{
    for(int i = 0; i < emigration_import.size(); i++)
    {
        recv_buffer_.push_back(emigration_import[i]);
        if(recv_buffer_.size() > buffer_capacity_)
            recv_buffer_.erase(recv_buffer_.begin());
    }

    return 0;
}
int LIFO::SelectFromBuffer(Population &emigration_export, int emigration_num)
{
    for(int i = 0; i < emigration_num && recv_buffer_.size() > 0; i++)
    {
        emigration_export.push_back(recv_buffer_[recv_buffer_.size() - 1]);
        recv_buffer_.erase(recv_buffer_.end());
    }
    return 0;
}
