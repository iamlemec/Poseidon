HOST_ARCH = $(shell uname -m)
CUDA_PATH = /media/Liquid/programs/cuda
GCC_BINDIR = ${CUDA_PATH}/bindir
COMPILER_PATH = /opt/gcc-4.9.3/lib64/gcc/x86_64-fedoraunited-linux-gnu

COMMON_INCL = -I/usr/include -I/media/Liquid/programs/cuda/include

NVCC = nvcc
NV_INCL = ${COMMON_INCL}
NV_COMP_OPTS = -fno-strict-aliasing -DUNIX
NV_OPT_FLAGS = -O2
NV_FLAGS = --ptxas-options=-v $(NV_OPT_FLAGS) --compiler-options $(NV_COMP_OPTS) --compiler-bindir $(GCC_BINDIR)

CC = g++
CC_INCL = ${COMMON_INCL}
CC_LIBS = -L${COMPILER_PATH}/lib64 -L${CUDA_PATH}/lib64 -L/usr/lib -lcudart -lGL -lGLU -lX11 -lXi -lXmu -lGLEW -lglut
CC_OPT_FLAGS = -O2
CC_DEBUG_FLAGS = -g
CC_WARN_FLAGS = -Wall
CC_FLAGS = $(CC_OPT_FLAGS) $(CC_WARN_FLAGS) -m64 -fPIC -fno-strict-aliasing -DUNIX

CU_TARGETS = Poseidon_cu
H_FILES = Poseidon_kernel.h

%.o : %.cpp $(H_FILES)
	$(CC) $(CC_FLAGS) $(CC_INCL) -c -o $@ $<

%.o : %.cu $(H_FILES)
	$(NVCC) $(NV_FLAGS) $(NV_INCL) -c -o $@ $<

Poseidon: Poseidon.o Poseidon_kernel.o
	$(CC) $(CC_FLAGS) $(CC_LIBS) -o $@ $^

test: test.o test_kernel.o
	$(CC) $(CC_FLAGS) $(CC_LIBS) -o $@ $^

all: Poseidon test

clean:
	rm -f *.o
	rm -f Poseidon
	rm -f test
