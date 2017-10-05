
OBJECTSD = decodeUtil.d.o huffdata.d.o jumptableapproach.d.o linapproach.d.o mainrun.d.o timing.d.o onethread.cud.o fastgpu.cud.o fastgpuOpt1.cud.o pes.d.o openclapproach.ocd.o
OBJECTSR = decodeUtil.r.o huffdata.r.o jumptableapproach.r.o linapproach.r.o mainrun.r.o timing.r.o onethread.cur.o fastgpu.cur.o fastgpuOpt1.cur.o pes.r.o openclapproach.ocr.o
TARGET = -DCUDA


all : HuffFramework HuffFrameworkDebug

%.d.o : ../%.c makefile
	gcc -g -O0 -Wall $(TARGET) -DFGPUDEBUG -c $< -o $@

%.r.o : ../%.c makefile
	gcc -O3 -Wall $(TARGET) -c $< -o $@

%.cud.o : ../%.cu makefile
	nvcc -g -O0 $(TARGET) -DFGPUDEBUG -c $< -o $@

%.ocd.o : ../%.c makefile
	gcc -g -O0 -Wall -Wno-deprecated-declarations -std=gnu99 $(TARGET) -DFGPUDEBUG -c $< -o $@

%.ocr.o : ../%.c makefile
	gcc -O3 -Wall -Wno-deprecated-declarations -std=gnu99 $(TARGET) -D_FORCE_INLINES -c $< -o $@

%.cur.o : ../%.cu makefile
	nvcc -O3 $(TARGET) -D_FORCE_INLINES -c $< -o $@

HuffFrameworkDebug : $(OBJECTSD) makefile
	gcc -g -O0 -Wall $(TARGET) -DFGPUDEBUG -o HuffFrameworkDebug $(OBJECTSD) -lcuda -lcudart -lOpenCL -lm -lstdc++

HuffFramework : $(OBJECTSR) makefile
	gcc -O3 -Wall $(TARGET) -o HuffFramework $(OBJECTSR) -lcuda -lcudart -lOpenCL -lm -lstdc++

clean :
	rm *.o  HuffFramework HuffFrameworkDebug 
