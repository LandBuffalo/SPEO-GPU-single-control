nvcc -rdc=true  -c -o tmp_kernel_DE_CUDA.o kernel_DE_CUDA.cu
nvcc -rdc=true  -c -o tmp_kernel_EA_CUDA.o kernel_EA_CUDA.cu

nvcc -dlink -o libkernel_DE_CUDA.o tmp_kernel_DE_CUDA.o
nvcc -dlink -o libkernel_EA_CUDA.o tmp_kernel_EA_CUDA.o
module load gcc/5.5.0
g++ -c  EA_CUDA.cc -lcudart -lcurand -I/apps/skylake/software/core/cuda/9.0.176/include -L/apps/skylake/software/core/cuda/9.0.176/lib64
g++ -c  DE_CUDA.cc -lcudart -lcurand -I/apps/skylake/software/core/cuda/9.0.176/include -L/apps/skylake/software/core/cuda/9.0.176/lib64

ar cur libkernel_DE_CUDA.a DE_CUDA.o tmp_kernel_DE_CUDA.o  libkernel_DE_CUDA.o
ar cur libkernel_EA_CUDA.a EA_CUDA.o tmp_kernel_EA_CUDA.o  libkernel_EA_CUDA.o

mv -f libkernel_DE_CUDA.a ../
mv -f libkernel_EA_CUDA.a ../

rm -f *.o
