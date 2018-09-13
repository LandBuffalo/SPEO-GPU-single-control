#!/bin/bash

module load cuda/9.0.176
cd ./CUDA_EA_library
bash make_CEC2014.sh
bash make_CEC2014_CUDA.sh
bash make_EA.sh
cd ..
mv libkernel_EA_CUDA.a ./src
mv libkernel_DE_CUDA.a ./src
mv libkernel_CEC2014_CUDA.a ./src
mv libkernel_CEC2014.a ./src
module load gcc/5.5.0
module load openmpi/3.0.0
make clean
make
cd ./bin
