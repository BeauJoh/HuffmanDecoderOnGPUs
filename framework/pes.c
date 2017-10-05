/*
 * pes.c  Parallel approach executed on a single cpu (serial)
 *
 *  Created on: 05/Sep/2016
 *      Author: ericm
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "decodeUtil.h"

#define KERNELSTART int bit; for (bit = 0; bit < bitsincomp; bit++) {
#define KERNELEND } 

void myerror(char *str) {
	fprintf(stderr, "error %s\n", str);
	exit(1);
}

void initbitsindex(int *bitsindex, int bitsincomp) {

	KERNELSTART
		bitsindex[bit] = -1;
	KERNELEND

}

void decodeAllBits(int bitsincomp, struct HuffNode *table, unsigned char *data,
		unsigned char *bitdecode, int *bitsteps) {
	int pos, tablepos, nextbit;
	KERNELSTART

		pos = bit;
		tablepos = 0;
		while ((table[tablepos].izero != -1) && pos < bitsincomp) {
			nextbit = (data[pos / 8] >> (pos % 8)) & 1;
			tablepos = (nextbit ? table[tablepos].ione : table[tablepos].izero);
			pos++;
		}
		bitdecode[bit] = table[tablepos].sym;
		bitsteps[bit] = pos - bit;

	KERNELEND
}

void makebigtable(int bitsincomp, struct HuffNode *table, unsigned char *data,
		unsigned char *bitdecode, int *bitsteps, int step, int powertwo,
		int *bitsteoresult) {
	int s, w;
	KERNELSTART

		s = bitsteps[bitsincomp * step + bit];
		if (s == -1 || bit + s > bitsincomp) {
			bitsteps[bitsincomp * (step + 1) + bit] = -1;
		} else {
			w = bitsteps[bitsincomp * step + bit + s];
			if (w == -1 || bit + s + w > bitsincomp) {
				bitsteps[bitsincomp * (step + 1) + bit] = -1;
			} else {
				bitsteps[bitsincomp * (step + 1) + bit] = s + w;
			}
		}

		//bitsteps[bitsincomp * (step + 1) + bit] = ((s == -1 || bit + s > bitsincomp) ? -1  : (((w= bitsteps[bitsincomp * step + bit + s]) == -1 || bit + s + w > bitsincomp)?-1:s+w));

	KERNELEND

	*bitsteoresult = bitsteps[bitsincomp * step + 0];
}

void calcbitsindex(int bitsincomp, int *bitsindex, int *bitsteps, int step,
		int powertwo) {
	KERNELSTART

		int offset = bitsteps[bitsincomp * (step - 1) + bit];
		int curval = bitsindex[bit];
		if (offset != -1 && curval != -1 && bit + offset < bitsincomp) {
			bitsindex[bit + offset] = curval + powertwo;

		}

	KERNELEND
}

void calcresult(int bitsincomp, int *bitsindex, unsigned char *bitdecode,
		unsigned char *result) {
	KERNELSTART

		if (bitsindex[bit] != -1) {
			result[bitsindex[bit]] = bitdecode[bit];
		}

	KERNELEND
}

void findmax(int bitsincomp, int *bitsindex) {
	int bit = bitsincomp - 1;
	while (bit > 0 && bitsindex[bit] == -1)
		bit--;
	bitsindex[0] = bitsindex[bit];

}

void pesApproach(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {

	int resultsize;
	int bitsincomp = cd->bits;

	int *bitsindex;
	struct HuffNode *table;
	unsigned char* data;
	int *bitsteps;
	unsigned char *bitdecode, *result;

	bitsindex = malloc(bitsincomp * sizeof(int));
	if (!bitsindex) {
		myerror("malloc - bitsindex");
	}
	initbitsindex(bitsindex, bitsincomp);

	table = cd->tree;

//	int datasize = (
//			(bitsincomp % 8) == 0 ? bitsincomp / 8 : (bitsincomp / 8) + 1);

	data = cd->data;
	result = uncompressed->data;
	bitsteps = malloc(25 * bitsincomp * sizeof(int));
	if (!bitsteps) {
		myerror("malloc - bitsteps");
	}

	bitdecode = malloc(bitsincomp * sizeof(unsigned char));
	if (!bitdecode) {
		myerror("malloc - bitdecode");
	}

#ifdef FGPUDEBUG
	printf("Memory Allocated\n");
#endif
	decodeAllBits(bitsincomp, table, data, bitdecode, bitsteps);

	int step = 0;
	int powertwo = 1;

	int bitsteoresult = 1;

	do {
#ifdef FGPUDEBUG
		printf("PES Make Step Tree step %d %d\n", step, bitsteoresult);
#endif

		makebigtable(bitsincomp, table, data, bitdecode, bitsteps, step,
				powertwo, &bitsteoresult);

        step++;
		powertwo = powertwo << 1;
	} while (bitsteoresult != -1);

#ifdef FGPUDEBUG
    int j,i;
	for (j=0;j<5;j++) {
		for (i=0;i<15;i++) {

			printf("  %3d ",bitsteps[j*bitsincomp + i]);
		}
		printf("\n");
	}
#endif

	powertwo = powertwo >> 1;
	bitsindex[0] = 0;

#ifdef FGPUDEBUG
	printf("calcbits index\n");
#endif

	while (step > 0) {
		calcbitsindex(bitsincomp, bitsindex, bitsteps, step, powertwo);
		step--;
		powertwo = powertwo >> 1;
	}

#ifdef FGPUDEBUG
	printf("calc result\n");
#endif

	//work out result
	calcresult(bitsincomp, bitsindex, bitdecode, result);

#ifdef FGPUDEBUG
	printf("find max\n");
#endif

	int maxvalue;
	findmax(bitsincomp, bitsindex);
	maxvalue = bitsindex[0];

	// obtain result from GPU
	resultsize = maxvalue + 1;
	uncompressed->uncompressedsize = resultsize;

	free(bitsindex);
	free(bitsteps);
	free(bitdecode);
}

