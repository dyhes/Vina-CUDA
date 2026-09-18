# Need to be modified according to different users
WORK_DIR=$(CURDIR)
BOOST_LIB_PATH=/usr
NVCC_COMPILER=/usr/local/cuda/bin/nvcc
DOCKING_BOX_SIZE=-DSMALL_BOX
GRID_DIM=-DGRID_DIM1=64 -DGRID_DIM2=128


# Should not be modified
BOOST_INC_PATH=-I$(BOOST_LIB_PATH)/include
VINA_GPU_INC_PATH=-I$(WORK_DIR)/lib -I$(WORK_DIR)/inc/ -I$(WORK_DIR)/inc/cuda
OPENCL_INC_PATH=
LIB1=-lboost_program_options -lboost_system -lboost_filesystem -lboost_thread
LIB2=-lstdc++ -lstdc++fs
LIB3=-lm -lpthread
LIB_PATH=-L$(BOOST_LIB_PATH)/lib/x86_64-linux-gnu
SRC=./lib/*.cpp
SRC_CUDA = ./inc/cuda/kernel1.cu ./inc/cuda/kernel2.cu ./inc/cuda/ric_gpu.cu ../../milestones/gpu_timer.cu
MACRO= $(GRID_DIM) $(DOCKING_BOX_SIZE) -Xcompiler -fopenmp -DBOOST_TIMER_ENABLE_DEPRECATED -DREF_GPU -DHOST_PROBE -lineinfo -arch=sm_89 -maxrregcount=40

all:out
out:./main/main.cpp
	$(NVCC_COMPILER) -o QuickVina2-GPU-2-1-CUDA $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp -O3 $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) -Xcompiler -fopenmp $(LIB_PATH) $(MACRO) $(OPTION) -DNDEBUG
source:./main/main.cpp
	$(NVCC_COMPILER) -o QuickVina2-GPU-2-1-CUDA $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp -O3 $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) -Xcompiler -fopenmp $(LIB_PATH) $(MACRO) $(OPTION) -DNDEBUG -DBUILD_KERNEL_FROM_SOURCE
debug:./main/main.cpp
	$(NVCC_COMPILER) -o QuickVina2-GPU-2-1-CUDA $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp -g $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) -Xcompiler -fopenmp $(LIB_PATH) $(MACRO) $(OPTION) -DBUILD_KERNEL_FROM_SOURCE
clean:
	rm -f QuickVina2-GPU-2-1-CUDA
