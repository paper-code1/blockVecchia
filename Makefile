#----------------complier configurations-------------------------
_SUPPORT_VECCHIA_?=TRUE
_USE_MAGMA_?=TRUE
#specify cuda directory
_CUDA_ROOT_=$(CUDA_HOME)
_CUDA_ARCH_ ?= 70
# specify compilers
CXX ?= g++
CC ?= gcc
NVCC=$(_CUDA_ROOT_)/bin/nvcc
NVOPTS = -ccbin $(CXX) --compiler-options -fno-strict-aliasing
COPTS = -fopenmp

NVOPTS_3 = -DTARGET_SM=$(_CUDA_ARCH_) -allow-unsupported-compiler -arch sm_$(_CUDA_ARCH_) -Xcompiler -fopenmp
ifdef _DEBUG_
  COPTS += -g -Xcompiler -rdynamic
  NVOPTS += -G -g -lineinfo
else
  COPTS += -O3
  NVOPTS += -O3
endif
ifdef _USE_MAGMA_
  COPTS += -DUSE_MAGMA
  _MAGMA_ROOT_?=$(HOME)/dev/magma-2.7.2
  NVOPTS += -DUSE_MAGMA
endif

#----------------complier configurations-------------------------


# include and lib paths
## our peacock server
INCLUDES=
INCLUDES+= -I.
INCLUDES+= -I${CUDA_ROOT}/include
INCLUDES+= -I${NLOPT_ROOT}/include
INCLUDES+= -I${GSL_ROOT}/include
INCLUDES+= -I./include

ifdef _USE_MAGMA_
	INCLUDES+= -I$(_MAGMA_ROOT_)/include
endif

LIB_PATH=
LIB_PATH+= -L${CUDA_ROOT}/lib64
LIB_PATH+= -L${NLOPT_ROOT}/lib
LIB_PATH+= -L${GSL_ROOT}/lib

ifdef _USE_MAGMA_
	LIB_PATH+= -L${_MAGMA_ROOT_}/lib
endif

# libraries to link against
LIB= -lm 
LIB+= -lnlopt  -lgsl
ifdef _USE_MAGMA_
	LIB+= -lmagma -lcusparse
endif
LIB+= -lcublas -lcudart
LIB+= -lgomp
LIB+= -lstdc++

INCLUDE_DIR=./include
OBJ_DIR=./obj
BIN_DIR=./bin
VECCHIA_BATCH=./src
include $(VECCHIA_BATCH)/Makefile

all: $(EXE_VECCHIA)

$(EXE_VECCHIA): $(BIN_DIR)/%: $(OBJ_DIR)/%.o $(OBJ_DIR)/cluster.o $(OBJ_DIR)/ckernel.o
	$(CC) $(COPTS) $(OBJ_DIR)/cluster.o $(OBJ_DIR)/ckernel.o $< -o $@ $(LIB_PATH) $(LIB)

$(OBJ_DIR)/cluster.o: $(VECCHIA_BATCH)/cluster.c
	$(CC) $(COPTS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/ckernel.o: $(VECCHIA_BATCH)/ckernel.cpp
	$(CXX) $(COPTS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o $(EXE_VECCHIA)
