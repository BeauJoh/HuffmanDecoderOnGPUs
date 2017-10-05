UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CCLIBS += -framework OpenCL
else
	CCLIBS += -lOpenCL -lm 
endif

DEBUG_CFLAGS += -g -O0 -Wall -Wno-deprecated-declarations -std=gnu99 -DOPENCL -DDEBUG -DBUILD_BINARY_KERNELS
RELEASE_CFLAGS += -O3 -Wall -Wno-deprecated-declarations -std=gnu99 -DOPENCL -DUSE_SOURCE_KERNELS
CC = gcc

OBJECTSD = decodeUtil.d.o huffdata.d.o pes.d.o jumptableapproach.d.o linapproach.d.o mainrun.d.o timing.d.o onethread.d.o openclapproach.d.o
OBJECTSR = decodeUtil.r.o huffdata.r.o pes.r.o jumptableapproach.r.o linapproach.r.o mainrun.r.o timing.r.o onethread.r.o openclapproach.r.o

all : HuffFrameworkCPU HuffFrameworkDebugCPU HuffFrameworkGPUBuildBinaryKernels HuffFrameworkGPUWithBinaryKernels HuffFrameworkGPUWithSourceKernels HuffFrameworkDebugGPU HuffFramework HuffFrameworkDebug

%.d.o : ../%.c makefile
	$(CC) $(DEBUG_CFLAGS) -Wall -c $< -o $@

%.r.o : ../%.c makefile
	$(CC) $(RELEASE_CFLAGS) -c $< -o $@

HuffFrameworkDebugCPU : $(OBJECTSD) makefile
	$(CC) $(DEBUG_CFLAGS) -DCPU_BUILD -o HuffFrameworkDebugCPU $(OBJECTSD) $(CCLIBS)

HuffFrameworkCPU : $(OBJECTSR) makefile
	$(CC) $(RELEASE_CFLAGS) -DCPU_BUILD -o HuffFrameworkCPU $(OBJECTSR) $(CCLIBS)

HuffFrameworkDebugGPU : $(OBJECTSD) makefile
	$(CC) $(DEBUG_CFLAGS) -DGPU_BUILD -o HuffFrameworkDebugGPU $(OBJECTSD) $(CCLIBS)

HuffFrameworkGPUBuildBinaryKernels : $(OBJECTSR) makefile
	$(CC) $(RELEASE_CFLAGS) -DGPU_BUILD -o HuffFrameworkGPUBuildBinaryKernels -DBUILD_BINARY_KERNELS -DUSE_BINARY_KERNELS $(OBJECTSR) $(CCLIBS)

HuffFrameworkGPUWithBinaryKernels : $(OBJECTSR) makefile
	$(CC) $(RELEASE_CFLAGS) -DGPU_BUILD -o HuffFrameworkGPUWithBinaryKernels -DUSE_BINARY_KERNELS $(OBJECTSR) $(CCLIBS)

HuffFrameworkGPUWithSourceKernels : $(OBJECTSR) makefile
	$(CC) $(RELEASE_CFLAGS) -DGPU_BUILD -o HuffFrameworkGPUWithSourceKernels $(OBJECTSR) $(CCLIBS)

HuffFrameworkDebug : $(OBJECTSD) makefile
	$(CC) $(DEBUG_CFLAGS) -o HuffFrameworkDebug $(OBJECTSD) $(CCLIBS)

HuffFramework : $(OBJECTSR) makefile
	$(CC) $(RELEASE_CFLAGS) -o HuffFramework $(OBJECTSR) $(CCLIBS)

clean :
	rm *.o  HuffFrameworkCPU HuffFrameworkDebugCPU HuffFrameworkGPUBuildBinaryKernels HuffFrameworkGPUWithBinaryKernels HuffFrameworkGPUWithSourceKernels HuffFrameworkDebugGPU HuffFramework HuffFrameworkDebug
