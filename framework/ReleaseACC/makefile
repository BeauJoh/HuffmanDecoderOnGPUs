
OBJECTSD = 
OBJECTSR = decodeUtil.pr.o huffdata.pr.o  mainrunacc.pr.o timing.pr.o  pacc.pr.o

all : HuffFramework



%.pr.o : ../%.c makefile
	pgcc -acc -Minfo=accel -c $< -o $@



HuffFramework : $(OBJECTSR) makefile
	pgcc -acc -Minfo=accel -ta=tesla:managed  -o HuffFramework  $(OBJECTSR) 

	
clean :
	rm *.o  HuffFramework HuffFrameworkDebug *~
