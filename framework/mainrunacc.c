/*
 * mainrun.c
 *
 *  Created on: 03/12/2015
 *      Author: ericm
 */

#include<stdio.h>
#include<stdlib.h>
#include<error.h>
#include<string.h>


#include"timing.h"
#include"huffdata.h"
#include"decodeUtil.h"
#include"jumptableapproach.h"
#include"linapproach.h"
#include"onethread.h"
#include"fastgpu.h"
#include"pes.h"
#include"pacc.h"

void readDataByte(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {
	int posb;
	int sum = 0;
	for (posb = 0; posb < cd->bits / 8; posb++) {
		sum += cd->data[posb];
	}
	uncompressed->data[0] = sum;
}

void simpleDecode(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {
	int pos = 0;
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
	}
}

void simpleDecodeByte(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {
	int pos = 0;
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
	}
}

void simpleDecoderp(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {

	int pos = 0;
	int nextbit;
	int tablepos = 0;
	int resultpos = 0;
	unsigned char b = 0;
	int bits = cd->bits;
	int i;
	while (pos < bits - 8) {
		b = cd->data[pos / 8];

		for (i = 0; i < 8; i++) {
			nextbit = (b >> (pos % 8)) & 1;
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
	b = cd->data[pos / 8];
	while (pos < bits) {

		nextbit = (b >> (pos % 8)) & 1;
		tablepos =
				(nextbit ? cd->tree[tablepos].ione : cd->tree[tablepos].izero);
		if (cd->tree[tablepos].ione == -1 && cd->tree[tablepos].izero == -1) { // its a leaf
			uncompressed->data[resultpos] = cd->tree[tablepos].sym;
			resultpos++;
			tablepos = 0;
		}
		pos++;
	}
}

void lookupsymbol(struct HuffNode *tree, unsigned char *data,
		unsigned char *sym, int *bits) {
	int pos = 0;
	int nextbit;
	int tablepos = 0;
	while (1) {
		nextbit = (data[pos / 8] >> (pos % 8)) & 1;
		tablepos = (nextbit ? tree[tablepos].ione : tree[tablepos].izero);
		if (tree[tablepos].ione == -1) { // its a leaf
			*sym = tree[tablepos].sym;
			*bits = pos + 1;
			return;
		}
		pos++;
	}
	error(1, 1, "lookupsymbol problem");
}

int mask2[] = { 0x000000, 0x000001, 0x000003, 0x000007, 0x00000F, 0x00001F,
		0x00003F, 0x00007F, 0x0000FF, 0x0001FF, 0x0003FF, 0x0007FF, 0x000FFF,
		0x001FFF, 0x003FFF, 0x007FFF, 0x00FFFF, 0x01FFFF, 0x03FFFF, 0x07FFFF,
		0x0FFFFF, 0x1FFFFF, 0x3FFFFF, 0x7FFFFF };

void decodeBigtablev1(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {

	int h = tableHeight(cd->tree, 0);
	int tablesize = 1 << h;

	//printf("%d\n",h);

	unsigned short *table;
	table = (unsigned short *) malloc(tablesize * sizeof(unsigned short));

	int i;
	for (i = 0; i < tablesize; i++) {
		unsigned char sym;
		int bits;
		lookupsymbol(cd->tree, (unsigned char *) &i, &sym, &bits);
		table[i] = (sym << 8) | bits;

		//printf("%d %c %d %X\n",i,sym,bits, table[i]);
	}

	//printf("%d %d",h, tablesize);

	int pos = 0;
	int index;
	int resultpos = 0;
	while (pos < cd->bits) {

		/*	index = 0;
		 for (i = 0; i < h; i++) {
		 nextbit = (cd->data[(pos + i) / 8] >> ((pos + i) % 8)) & 1;
		 index |= nextbit << i;
		 }*/

		index = ((cd->data[pos >> 3] | ((cd->data[(pos >> 3) + 1]) << 8)
				| ((cd->data[(pos >> 3) + 2]) << 16)
				| ((cd->data[(pos >> 3) + 3]) << 24)) >> (pos & 7)) & mask2[h];

		int lookupval;
		lookupval = table[index];
		unsigned char sym = (lookupval >> 8) & 0xFF;
		int bits = lookupval & 0xFF;

		uncompressed->data[resultpos] = sym;
		resultpos++;
		// printf("%X index %d sym  %d bits %d\n",lookupval, index,sym, bits);

		//printf("%c",sym);

		pos += bits;

	}
	free(table);
}

struct bigTableMulti {
	unsigned char bitsused;
	unsigned char numSym;
	unsigned char syms[6];
};

void lookupsymbols(struct HuffNode *tree, unsigned char *data,
		struct bigTableMulti *te, int h) {
	int pos = 0;
	int nextbit;
	int tablepos = 0;
	te->numSym = 0;
	while (pos < h && te->numSym < 6) {
		nextbit = (data[pos / 8] >> (pos % 8)) & 1;
		tablepos = (nextbit ? tree[tablepos].ione : tree[tablepos].izero);
		if (tree[tablepos].ione == -1) { // its a leaf
			te->syms[te->numSym] = tree[tablepos].sym;
			te->numSym++;
			te->bitsused = pos + 1;
			tablepos = 0;
		}
		pos++;
	}

}

void decodeBigtableMultiSym(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {

	int h = tableHeight(cd->tree, 0);
	int tablesize = 1 << h;

	struct bigTableMulti *table;
	table = (struct bigTableMulti *) malloc(
			tablesize * sizeof(struct bigTableMulti));

	int i;
	for (i = 0; i < tablesize; i++) {
		lookupsymbols(cd->tree, (unsigned char *) &i, &(table[i]), h);
	}
#ifdef DEBUG
	printf("%d %d\n",h, tablesize);
	for (i = 0; i < tablesize; i++) {
		printf("%d %d : %d %d\n", i,h, table[i].numSym , table[i].bitsused );
	}
#endif

	int pos = 0;
	int index;
	int nextbit;
	int resultpos = 0;
	while (pos < cd->bits - h) {
		int i;
		index = ((cd->data[pos >> 3] | ((cd->data[(pos >> 3) + 1]) << 8)
				| ((cd->data[(pos >> 3) + 2]) << 16)
				| ((cd->data[(pos >> 3) + 3]) << 24)) >> (pos & 7)) & mask2[h];

		for (i = 0; i < table[index].numSym; i++) {
			uncompressed->data[resultpos] = table[index].syms[i];
			resultpos++;
		}
		pos += table[index].bitsused;

	}
	int tablepos = 0;
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
	}

	free(table);
}

void showbits(unsigned int v) {
	int i;
	for (i = 0; i < 32; i++)
		printf("%d", (v >> i & 1));
	printf("\n");
}

void setTargetSizes(struct CompressedData *tcd, int targetsize) {

	int pos = 0;
	int nextbit;
	int tablepos = 0;
	int resultpos = 0;

	int lastokaypos = 0;

	while (pos < targetsize) {
		nextbit = (tcd->data[pos / 8] >> (pos % 8)) & 1;
		tablepos = (
				nextbit ? tcd->tree[tablepos].ione : tcd->tree[tablepos].izero);
		if (tcd->tree[tablepos].ione == -1 && tcd->tree[tablepos].izero == -1) { // its a leaf
		//uncompressed->data[resultpos] = cd->tree[tablepos].sym;
			resultpos++;
			tablepos = 0;
			lastokaypos = pos;
		}
		pos++;
	}

	tcd->bits = lastokaypos + 1;
	tcd->uncompressedsize = resultpos;
}

void graphtest(struct decoder *d, struct TestData *td, int incs) {
	int testsize = incs;
	struct CompressedData reducedCompdata;
	reducedCompdata.data = td->cd->data;
	reducedCompdata.nodes = td->cd->nodes;
	reducedCompdata.tree = td->cd->tree;

	struct UnCompressedData reducedUnCompressed;

	reducedUnCompressed.data = td->ucd->data;
	struct TestData reducedTd;
	reducedTd.name = td->name;
	reducedTd.cd = &reducedCompdata;
	reducedTd.ucd = &reducedUnCompressed;

	while (testsize < td->cd->bits) {
		setTargetSizes(&reducedCompdata, testsize);
		reducedUnCompressed.uncompressedsize = reducedCompdata.uncompressedsize;

		printf("%8d  %.9f\n", testsize, evaluate(d, &reducedTd, 1));
		testsize += incs;
	}

}

void evalandshow(struct decoder *d, struct TestData *td, int withcheck) {
	if (d->paramdata != NULL) {
		printf("%17s %8s  %2d %.9f\n", d->name, td->name,
				*((int *) d->paramdata), evaluate(d, td, withcheck));
	} else {
		printf("%17s %8s     %.9f\n", d->name, td->name,
				evaluate(d, td, withcheck));
	}
}

struct decoder *simpledec,  *pacc , *justreaddata;
int jumptabpar;

void testall(struct TestData *td) {
	int i;

	evalandshow(justreaddata, td, 0);
	evalandshow(simpledec, td, 1);
	evalandshow(pacc, td, 1);

}

void errorexit(char *str) {
	fprintf(stderr, "error exit: %s\n", str);
	exit(1);
}

int main(int argc, char *argv[]) {
	char *testname;
	//int parm1;

	if (argc > 1) {
		testname = argv[1];
		if (strcmp(argv[1], "hello")) {
			//parm1 = 1;
		}
	} else {
		testname = "default";
	}
	fprintf(stderr, "running test: %s\n", testname);

	//reportresolution();
	fprintf(stderr, "starting....\n");

    pacc = newDecoder(paccApproach, NULL, "pacc");

	justreaddata = newDecoder(readDataByte, NULL, "justreaddata");
	simpledec = newDecoder(simpleDecode, NULL, "simpleDecode");

	struct TestData *kjvtest, *hellotest, *book2test, *newstest, *paper1test;

	kjvtest = loadTestData("../../files/kjv.txt", "kjv");
	hellotest = loadTestData("../../files/hello", "hello");
	book2test = loadTestData("../../files/book2", "book2");
	newstest = loadTestData("../../files/news", "news");
	paper1test = loadTestData("../../files/paper1", "paper1");

	//showDataBits(hellotest->cd);
	if (strcmp(testname, "default") == 0) {
		showHuffTree(hellotest->cd->tree, 0);
		listHuffCodes(hellotest->cd->tree, 0);
		showHuffTable(hellotest->cd->tree, hellotest->cd->nodes);
		printf(" tablenodes : %d\n", treeSize(hellotest->cd->tree, 0));
		printf("tablegroups  1 : %d \n",
				tableNumGroups(hellotest->cd->tree, 1, 0));
		printf("tablegroups  2 : %d \n",
				tableNumGroups(hellotest->cd->tree, 2, 0));
		printf("tablegroups  3 : %d \n",
				tableNumGroups(hellotest->cd->tree, 3, 0));
		printf("tablegroups  4 : %d \n",
				tableNumGroups(hellotest->cd->tree, 4, 0));
		printf("%d\n", tableNumGroups(hellotest->cd->tree, 4, 0));
	} else if (strcmp(testname, "hello") == 0) {
		evalandshow(simpledec, hellotest, 1);
		evalandshow(pacc, hellotest, 1);
	} else if (strcmp(testname, "kjv") == 0) {

	    evalandshow(pacc, kjvtest, 1);

	} else if (strcmp(testname, "bigtable") == 0) {
		infoTestData(paper1test);
		infoTestData(hellotest);
		infoTestData(newstest);
		infoTestData(kjvtest);
		infoTestData(book2test);

		evalandshow(pacc, paper1test, 1);
		evalandshow(pacc, hellotest, 1);
		evalandshow(pacc, newstest, 1);
		evalandshow(pacc, kjvtest, 1);
		evalandshow(pacc, book2test, 1);






	} else if (strcmp(testname, "graph3") == 0) {
		graphtest(pacc, kjvtest, 500000);
	}  else if (strcmp(testname, "kjvprof") == 0) {
		evalandshow(pacc, kjvtest, 1);
	}



	//graphtest(simpledec, kjvtest, 100000);
	/*	jumptabpar = 8;
	 evalandshow(linapproach, book2test, 1);
	 evalandshow(linapproach, book2test, 1);
	 evalandshow(linapproach, book2test, 1);
	 */
//	printf("starting kjv main run\n");
//	evalandshow(linapproach, kjvtest, 1);
	//testall(newstest);
	//testall(paper1test);
	//testall(book2test);
	//testall(hellotest);
	//testall(kjvtest);
	/*
	 printf("%s : %f\n", dbtv2->name, evaluate(dbtv2, kjvtest));

	 printf("%s : %f\n", simpledec->name, evaluate(simpledec, kjvtest));

	 printf("%s : %f\n", simpledecbyte->name, evaluate(simpledecbyte, kjvtest));
	 printf("%s : %f\n", simplerp->name, evaluate(simplerp, kjvtest));
	 */

	freeTestData(hellotest);
	freeTestData(kjvtest);
	freeTestData(book2test);
	freeTestData(newstest);
	freeTestData(paper1test);

	freeDecoder(simpledec);


	freeDecoder(justreaddata);
	freeDecoder(pacc);


	return 0;
}

/*
 * Hello world encoding:
 * 110 0000 01 01 001 100 0001 001 101 01 111
 * H   e    l  l  o   ' ' W    o   r   l  d
 */

