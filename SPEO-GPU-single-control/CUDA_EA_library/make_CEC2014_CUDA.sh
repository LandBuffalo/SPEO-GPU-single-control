nvcc -rdc=true  -c -o tmp_kernel_CEC2014_CUDA.o kernel_CEC2014_CUDA.cu

nvcc -dlink -o libkernel_CEC2014_CUDA.o tmp_kernel_CEC2014_CUDA.o -lcudart
module load gcc/5.5.0
g++ -c  CEC2014_CUDA.cc -lcudart -I/apps/skylake/software/core/cuda/9.0.176/include -L/apps/skylake/software/core/cuda/9.0.176/lib64

ar cur libkernel_CEC2014_CUDA.a CEC2014_CUDA.o tmp_kernel_CEC2014_CUDA.o  libkernel_CEC2014_CUDA.o

mv -f libkernel_CEC2014_CUDA.a ../
rm -f *.o
