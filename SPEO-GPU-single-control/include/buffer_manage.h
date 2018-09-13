#ifndef __BUFFER_MANAGE__
#define __BUFFER_MANAGE__

#include "random.h"
#include <algorithm>

class BufferManage
{
protected:
    Population          recv_buffer_;
    int                 buffer_capacity_;
    Random              random_;
public:
                        BufferManage();
                        BufferManage(int buffer_capacity);
    virtual             ~BufferManage();
    virtual int         UpdateBuffer(Population &emigration_import) = 0;
    virtual int         SelectFromBuffer(Population &emigration_export, int emigration_num) = 0;
    virtual real      CalDiversity();
};

class OnlineCluster: public BufferManage
{
protected:
    Population          cluster_centre_;
    vector<int>         cluster_member_num_;
    vector<int>         flag_updated_;
    int                 FindNearestIndividual(Individual &individual, Population &population);
    real              CalDistance(Individual &individual1, Individual &individual2);
    virtual int         ReplaceMemberToCluster(int belong_cluster_ID, Individual &individual);
    int                 CreatCluster(Individual &individual);
public:
                        OnlineCluster(int buffer_capacity);
    virtual             ~OnlineCluster();
    virtual int         UpdateBuffer(Population &emigration_import);
    int                 SelectFromBuffer(Population &emigration_export, int emigration_num);
    virtual real      CalDiversity();
};

class BestSelect : public BufferManage
{
private:
    int                 FindBestBufferMember();
public:
                        BestSelect(int buffer_capacity);
    virtual             ~BestSelect();
    virtual int         UpdateBuffer(Population &emigration_import);
    virtual int         SelectFromBuffer(Population &emigration_export, int emigration_num);
};

class RandomSelect : public BufferManage
{
private:
public:
                        RandomSelect(int buffer_capacity);
    virtual             ~RandomSelect();
    virtual int         UpdateBuffer(Population &emigration_import);
    virtual int         SelectFromBuffer(Population &emigration_export, int emigration_num);
};

class FIFO : public BufferManage
{
private:
public:
                        FIFO(int buffer_capacity);
    virtual             ~FIFO();
    virtual int         UpdateBuffer(Population &emigration_import);
    virtual int         SelectFromBuffer(Population &emigration_export, int emigration_num);
};

class LIFO : public BufferManage
{
private:
public:
                        LIFO(int buffer_capacity);
    virtual             ~LIFO();
    virtual int         UpdateBuffer(Population &emigration_import);
    virtual int         SelectFromBuffer(Population &emigration_export, int emigration_num);

};
#endif
