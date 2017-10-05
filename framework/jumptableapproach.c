/*
 * jumptableapproach.c
 *
 *  Created on: 08/12/2015
 *      Author: ericm
 */

#include "error.h"
#include <stdlib.h>
#include <stdio.h>

#include "jumptableapproach.h"

void showjumptables(int jumpbits, int tablesize, struct tableIndex *indextab,
		struct jumpElement7 *jumptable, int *jumptableCurrSize) {
	printf("jumpbits %d tablesize %d jumptableCurrSize %d\n", jumpbits,
			tablesize, *jumptableCurrSize);

	int i, j;
	for (i = 0; i < *jumptableCurrSize; i++) {
		printf("T%d %d ", i, indextab[i].prebitsnum);
		showbits2(indextab[i].prebitsnum, indextab[i].prebits);
		printf("\n");

		for (j = 0; j < tablesize; j++) {
			showbits2(jumpbits, j);
			printf(" ");
			printf("%2d ", jumptable[tablesize * i + j].nextTable);
			int k;
			for (k = 0; k < jumptable[tablesize * i + j].numSym; k++) {
				printf("'%c'", jumptable[tablesize * i + j].syms[k]);
			}
			printf("\n");
		}

	}

}

int makejumptables(int jumpbits, int tablesize, struct tableIndex *indextab,
		struct jumpElement7 *jumptable, int prebits, int prebitsnum,
		int *jumptableCurrSize, struct HuffNode *tree, int root) {
	// find if one exist - and done if so
//printf("PREBITSNUM %d\n", prebitsnum );
	int i;
	for (i = 0; i < *jumptableCurrSize; i++) { // this could be done with a hash or just table lookup.
		if (indextab[i].prebits == prebits
				&& indextab[i].prebitsnum == prebitsnum) {
			//printf("SAME %d i %d\n", prebits,i );
			return i;
		}
	}

	int thisjump = *jumptableCurrSize;
	(*jumptableCurrSize)++;
	indextab[thisjump].prebits = prebits;
	indextab[thisjump].prebitsnum = prebitsnum;
	//printf("Setting %d PREBITSNUM %d\n", thisjump, prebitsnum );

	// add a new table and recure

	int j;

	for (i = 0; i < tablesize; i++) {

		int nextbit;
		int tablepos = root;

		int resultindex = tablesize * thisjump + i;
		int resultsymcount = 0;

		prebits = indextab[thisjump].prebits;
		prebitsnum = indextab[thisjump].prebitsnum;

		for (j = 0; j < jumpbits; j++) {
//			nextbit = (i >> ((jumpbits - 1) - j)) & 1;
			nextbit = (i >> j) & 1; // reverse the bits
			prebits = (prebits << 1) | nextbit;
			prebitsnum++;
			tablepos = (nextbit ? tree[tablepos].ione : tree[tablepos].izero);
			if (tree[tablepos].ione == -1) { // its a leaf
				jumptable[resultindex].syms[resultsymcount] =
						tree[tablepos].sym;
				resultsymcount++;
				prebitsnum = 0;
				prebits = 0;
				tablepos = 0;
			}

		}
		jumptable[resultindex].numSym = resultsymcount;
		jumptable[resultindex].nextTable = makejumptables(jumpbits, tablesize,
				indextab, jumptable, prebits, prebitsnum, jumptableCurrSize,
				tree, tablepos);

	}
	return thisjump;

}

unsigned char reverse(unsigned char b) {
	return (unsigned char) (((b * 0x0802U & 0x22110U) | (b * 0x8020U & 0x88440U))
			* 0x10101U >> 16);
}

int mask[] = { 0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F,
		0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF };

void jumptableApproach(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {
	/*int pos = 0;
	 int nextbit;
	 int tablepos = 0;
	 int resultpos = 0;
	 while (pos < cd->bits) {
	 nextbit = (cd->data[pos / 8] >> (pos % 8)) & 1;
	 tablepos =
	 (nextbit ? cd->tree[tablepos].ione : cd->tree[tablepos].izero);
	 if (cd->tree[tablepos].ione == -1 && cd->tree[tablepos].izero == -1) { // its a leaf
	 uncompressed->data[resultpos] = cd->tree[tablepos].sym;
	 resultpos++;
	 tablepos = 0;
	 }
	 pos++;
	 }*/

	int jumpbits, tablesize, mindepth, maxnumbertables, jumptableCurrSize,
			treesize;
	//int tabletype;

	jumpbits = *((int *) paramdata);
	//jumpbits = 4;
	tablesize = 1 << jumpbits;

	mindepth = tableMinDepth(cd->tree, 0);
//	height = tableHeight(cd->tree, 0);
	treesize = treeSize(cd->tree, 0);

	/*	if ((jumpbits / mindepth) <= 3) {
	 tabletype = 7; // change to 3 and duplicate code
	 } else if ((jumpbits / mindepth) <= 7) {
	 tabletype = 7;
	 } else {
	 error(1, 1, "tabletype problem");
	 }*/
	if ((jumpbits / mindepth) > 7)
		error(1, 1, "tabletype problem");

	maxnumbertables = (1 << jumpbits) - 1 + treesize;

	struct jumpElement7 *jumptable;
	jumptable = (struct jumpElement7 *) malloc(
			sizeof(struct jumpElement7) * maxnumbertables * tablesize);

	struct tableIndex *indextab;
	indextab = (struct tableIndex *) malloc(
			sizeof(struct tableIndex) * maxnumbertables);
	jumptableCurrSize = 0;

	makejumptables(jumpbits, tablesize, indextab, jumptable, 0, 0,
			&jumptableCurrSize, cd->tree, 0);

//	showjumptables(jumpbits, tablesize, indextab, jumptable,
//				&jumptableCurrSize);

	int pos = 0;
	int resultpos = 0;
	int currtable = 0;
	int index, nextbit, j;
	struct jumpElement7 *jumpelement;
	int limit = (cd->bits) >> 3;

	if (jumpbits == 8) {
		while (pos < limit) {

			//index = reverse(cd->data[pos>>3]);
			index = cd->data[pos];
			jumpelement = &(jumptable[tablesize * currtable + index]);
			for (j = 0; j < jumpelement->numSym; j++) {
				uncompressed->data[resultpos] = jumpelement->syms[j];
				resultpos++;
			}

			currtable = jumpelement->nextTable;
			pos += 1;

		}

		pos = pos << 3;
		int tablepos = 0;

		pos = pos - indextab[currtable].prebitsnum;
		while (pos < cd->bits) {
			nextbit = (cd->data[pos / 8] >> (pos % 8)) & 1;
			tablepos =
					(nextbit ?
							cd->tree[tablepos].ione : cd->tree[tablepos].izero);
			if (cd->tree[tablepos].ione == -1
					&& cd->tree[tablepos].izero == -1) { // its a leaf
				uncompressed->data[resultpos] = cd->tree[tablepos].sym;
				resultpos++;
				tablepos = 0;
			}
			pos++;
		}
	} else {
		if (jumpbits > 8) {
			while (pos < (cd->bits - jumpbits)) {

				index =
						((cd->data[pos >> 3] | ((cd->data[(pos >> 3) + 1]) << 8) | ((cd->data[(pos >> 3) + 2]) << 16))
								>> (pos & 7)) & mask[jumpbits];

				jumpelement = &(jumptable[tablesize * currtable + index]);
				for (j = 0; j < jumpelement->numSym; j++) {
					uncompressed->data[resultpos] = jumpelement->syms[j];
					resultpos++;
				}
				currtable = jumpelement->nextTable;

				pos += jumpbits;
			}
		} else {

			while (pos < (cd->bits - jumpbits)) {

				index =
						((cd->data[pos >> 3] | ((cd->data[(pos >> 3) + 1]) << 8))
								>> (pos & 7)) & mask[jumpbits];

				jumpelement = &(jumptable[tablesize * currtable + index]);
				for (j = 0; j < jumpelement->numSym; j++) {
					uncompressed->data[resultpos] = jumpelement->syms[j];
					resultpos++;
				}
				currtable = jumpelement->nextTable;

				pos += jumpbits;
			}

		}
		int tablepos = 0;

		pos = pos - indextab[currtable].prebitsnum;
		while (pos < cd->bits) {
			nextbit = (cd->data[pos / 8] >> (pos % 8)) & 1;
			tablepos =
					(nextbit ?
							cd->tree[tablepos].ione : cd->tree[tablepos].izero);
			if (cd->tree[tablepos].ione == -1
					&& cd->tree[tablepos].izero == -1) { // its a leaf
				uncompressed->data[resultpos] = cd->tree[tablepos].sym;
				resultpos++;
				tablepos = 0;
			}
			pos++;
		}
	}
//int w;
//for (w=0;w<uncompressed->uncompressedsize;w++ ) printf("%c", uncompressed->data[w]);
//printf("\n\n\n");

	free(jumptable);
	free(indextab);
}
