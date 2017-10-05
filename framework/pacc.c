/*
 * pes.c  Parallel approach executed on a single cpu (serial)
 *
 *  Created on: 05/Sep/2016
 *      Author: ericm
 */


// pgcc -acc -Minfo=accel -c pacc.pr.o ../pacc.c
// see http://www.openacc.org/sites/default/files/OpenACC_Programming_Guide_0.pdf

#include <stdio.h>
#include <stdlib.h>

#include "decodeUtil.h"



void myerror(char *str) {
	fprintf(stderr, "error %s\n", str);
	exit(1);
}



void paccApproach(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {

	int resultsize;
	int bitsincomp = cd->bits;

	int *bitsindex;
	struct HuffNode *table;
	unsigned char* data;
	int *bitsteps;
	unsigned char *bitdecode, *result;
	int wlogsize = 28;
	int nodecount = cd->nodes;

	int bitstepssize = wlogsize * bitsincomp;
printf("bitstepsize %d\n", bitstepssize);
	bitsindex = malloc(bitsincomp * sizeof(int));
	if (!bitsindex) {
		myerror("malloc - bitsindex");
	}

	bitsteps = malloc(bitstepssize * sizeof(int));
	if (!bitsteps) {
		myerror("malloc - bitsteps");
	}

	bitdecode = malloc(bitsincomp * sizeof(unsigned char));
	if (!bitdecode) {
		myerror("malloc - bitdecode");
	}

	data = cd->data;
	result = uncompressed->data;
	table = cd->tree;

	int datasize = (
				(bitsincomp % 8) == 0 ? bitsincomp / 8 : (bitsincomp / 8) + 1);



//#pragma acc data copyout(bitsindex[bitsincomp])
//#pragma acc data pcopyout(bitsteps[bitstepssize])
//#pragma acc data copyout(bitdecode[bitsincomp])
//#pragma acc data copyin(table[nodecount])
//#pragma acc data create(result[bitsincomp])
//#pragma acc data copyin(data[datasize])
#pragma acc data copyin(table[nodecount]) copyin(data[datasize])
{

#pragma acc parallel loop copyout(bitsindex[bitsincomp])
	for (int i = 0; i < bitsincomp; i++) {
		bitsindex[i] = -1;
	}




#pragma acc parallel loop copyout(bitsteps[bitstepssize])
	for (int i = 0; i < bitsincomp; i++) {
		int pos, tablepos, nextbit;  // local to device
		pos = i;
		tablepos = 0;
		while ((table[tablepos].izero != -1) && pos < bitsincomp) {
			nextbit = (data[pos / 8] >> (pos % 8)) & 1;
			tablepos = (nextbit ? table[tablepos].ione : table[tablepos].izero);
			pos++;
		}
		bitdecode[i] = table[tablepos].sym;
		bitsteps[i] = pos - i;

	}



	int step = 0;
	int powertwo = 1;
	int bitsteoresult = 1;

	do {

       // #pragma acc data pcopy(bitsteps[bitstepssize])

printf("step : %d\n", step);
		#pragma acc parallel loop copy(bitsteps[bitstepssize])
		for (int i = 0; i < bitsincomp; i++) {
			int s, w; // local
			s = bitsteps[(bitsincomp * step) + i];
			if (s == -1 || i + s > bitsincomp) {
				bitsteps[(bitsincomp * (step + 1)) + i] = -1;
			} else {
				w = bitsteps[(bitsincomp * step) + i + s];
				if (w == -1 || i + s + w > bitsincomp) {
					bitsteps[(bitsincomp * (step + 1)) + i] = -1;
				} else {
					bitsteps[(bitsincomp * (step + 1)) + i] = s + w;
				}
			}
		}

//#pragma acc data pcopyout(bitsteps[bitstepssize])

		bitsteoresult = bitsteps[bitsincomp * step + 0];
		printf("bitsteorresult : %d\n", bitsteoresult);

		step++;
		powertwo = powertwo << 1;
	} while (bitsteoresult != -1);






	powertwo = powertwo >> 1;
	bitsindex[0] = 0;


	while (step > 0) {
		
	//	#pragma acc parallel loop
		for (int i = 0; i < bitsincomp; i++) {

			int offset = bitsteps[bitsincomp * (step - 1) + i];
			int curval = bitsindex[i];
			if (offset != -1 && curval != -1 && i + offset < bitsincomp) {
				bitsindex[i + offset] = curval + powertwo;
			}
		}
		step--;
		powertwo = powertwo >> 1;
	}

	//work out result
	


	//#pragma acc parallel loop
	for (int i = 0; i < bitsincomp; i++) {
		if (bitsindex[i] != -1) {
			result[bitsindex[i]] = bitdecode[i];
		}
	}


}


	int maxvalue, bit;


	bit = bitsincomp - 1;
	while (bit > 0 && bitsindex[bit] == -1)
		bit--;
	maxvalue = bitsindex[bit];
	
	// obtain result from GPU
	resultsize = maxvalue + 1;
	uncompressed->uncompressedsize = resultsize;


	free(bitsindex);
	free(bitsteps);
	free(bitdecode);
}

