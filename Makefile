# C compiler
CC = g++
CC_FLAGS = -std=c++11 -g -O2

UNAME := $(shell uname)

OPEN_CL_LINK = -lOpenCL
$(info Uname: $(UNAME))

ifeq ($(UNAME),Darwin)
		OPEN_CL_LINK = -framework OpenCL
endif

$(info OpenCL Link Command is: $(OPEN_CL_LINK))


all:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $(OPEN_CL_LINK) alg.cpp -g -o alg
