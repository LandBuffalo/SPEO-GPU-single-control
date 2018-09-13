INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include  /apps/skylake/software/core/cuda/9.0.176/include)

link_directories(./ /apps/skylake/software/core/cuda/9.0.176/lib64)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(./ DIR_SRCS)
ADD_EXECUTABLE(SPEO_GPU ${DIR_SRCS})

target_link_libraries(SPEO_GPU mpi  kernel_EA_CUDA  kernel_DE_CUDA  kernel_CEC2014_CUDA kernel_CEC2014 cudart curand)

