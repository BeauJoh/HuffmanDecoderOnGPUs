/*
 * huffdata.c
 *
 *  Created on: 03/12/2015
 *      Author: ericm
 */

#include "huffdata.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include "byteswap.h"
#include "error.h"

#include "decodeUtil.h"

int readBint(FILE *f) {
	int d;
	assert(fread(&d, 4, 1, f) == 1);
	return bswap_32(d);
}

struct CompressedData *loadHuffFile(char filename[]) {
	struct CompressedData *res;
	res = (struct CompressedData *) malloc(sizeof(struct CompressedData));

	FILE *f;
	f = fopen(filename, "r");

	// read the header
	char header[4];
	assert(fread(header, 1, 4, f) == 4);
	if (header[0] != 'H' || header[1] != 'U' || header[2] != 'F'
			|| header[3] != 'F')
		error(1, 1,"HUFF expected");

	res->nodes = readBint(f);
	res->bits = readBint(f);
	res->uncompressedsize = readBint(f);

	// read the tree
	res->tree = (struct HuffNode *) malloc(
			sizeof(struct HuffNode) * res->nodes);
	int i;

	for (i = 0; i < res->nodes; i++) {
		assert(fread(&(res->tree[i].sym), 1, 1, f) == 1);
		res->tree[i].izero = readBint(f);
		res->tree[i].ione = readBint(f);
	}
	int compbytes = (res->bits % 8 == 0 ? res->bits / 8 : (res->bits / 8) + 1);

	// read the compressed data
	res->data = (unsigned char *) malloc(compbytes + 3);
	for (i = 0; i < compbytes; i++) {
		assert(fread(&(res->data[i]), 1, 1, f) == 1);
	}
	res->data[compbytes] = 0;
	res->data[compbytes + 1] = 0;
	res->data[compbytes + 2] = 0;

	fclose(f);
	return res;
}

void infoCompressedData(struct CompressedData *cd) {
	printf("nodes %d, bits %d, uncompressedsize %d\n", cd->nodes, cd->bits,
			cd->uncompressedsize);
}


void infoTestData(struct TestData *td) {
	printf("%s ", td->name);
	infoCompressedData(td->cd);
}

void freeCompressedData(struct CompressedData *cd) {
	free(cd->tree);
	free(cd->data);
	free(cd);

}

void space(int v) {
	int i;
	for(i=0;i<v;i++) printf(" ");
}

void printlevel(int lev, int wid, struct HuffNode *nodes, int root) {
	char str[10];
	if (root == -1) {
		space(wid);
	} else if (lev == 0) {
		if (nodes[root].izero == -1) {
		   sprintf(str,"%d'%c'", nodes[root].sym, nodes[root].sym);
		} else {
			sprintf(str,"*");
		}
		int slen = (int)strlen(str);
		int fspace = (wid - slen) / 2;
		space(fspace);
		printf("%s",str);
		space(wid-fspace-slen);
	} else {
		printlevel(lev-1, wid/2, nodes, nodes[root].izero);
		printlevel(lev-1, wid - (wid/2), nodes, nodes[root].ione);
	}

}

void showbits2(int len, int bits) {
	int i;
	for (i=0;i<len;i++) {
		printf("%s", (((bits >> ((len-1)-i)) & 1)?"1":"0"));
	}

}

void showHuffTree(struct HuffNode *nodes, int root) {
	int h = tableHeight(nodes, root);
	int w = 1 << (h+3);
	int i;
	for (i=0;i<h+1;i++) {
		printlevel(i,w,nodes,root);
		printf("\n");
	}
}

void listHuffCodesPre(struct HuffNode *nodes, int root, int prelen, int bits) {
    if (nodes[root].ione == -1) {
    	showbits2(prelen, bits);
    	printf(" '%c'\n",  nodes[root].sym);
    } else {
    	listHuffCodesPre(nodes, nodes[root].izero, prelen + 1, bits << 1);
    	listHuffCodesPre(nodes, nodes[root].ione, prelen + 1, (bits << 1) | 1 );

    }
}

void listHuffCodes(struct HuffNode *nodes, int root) {
	listHuffCodesPre(nodes, root, 0,0);
}





struct UnCompressedData *loadTextFile(char filename[]) {
	struct UnCompressedData *res;
	res = (struct UnCompressedData *) malloc(sizeof(struct UnCompressedData));
	FILE *f;
	f = fopen(filename, "r");
	struct stat buf;
	fstat(fileno(f), &buf);
	int size = (int)buf.st_size;
	res->uncompressedsize = size;
	res->data = (unsigned char *) malloc(size+3);
	assert(fread(res->data, size, 1, f) == 1);
	fclose(f);
	return res;
}
struct UnCompressedData *newUnCompressedData(int size) {
	struct UnCompressedData *res;
	res = (struct UnCompressedData *) malloc(sizeof(struct UnCompressedData));
	res->uncompressedsize = size;
	res->data = (unsigned char *) malloc(size+3);
	return res;

}
void freeUnCompressedData(struct UnCompressedData *ucd) {
	free(ucd->data);
	free(ucd);
}

void clearUnCompressedData(struct UnCompressedData *ucd) {
	memset(ucd->data, 0, ucd->uncompressedsize);
}

int compareUnCompressedData(struct UnCompressedData *ucd1, struct UnCompressedData *ucd2) {
	if (ucd1->uncompressedsize != ucd2->uncompressedsize) {
		printf("different size! : %d %d\n",ucd1->uncompressedsize, ucd2->uncompressedsize );
		return -1;
	}
	int i;
	int sumdiff = 0;
	for (i = 0; i < ucd1->uncompressedsize; i++) {
		if (ucd1->data[i] != ucd2->data[i]) {

			if (sumdiff <10) printf("different at: %d  val1: %d  val2: %d\n",i, ucd1->data[i], ucd2->data[i] );
			sumdiff++;
		}
	}
	if (sumdiff == 0) {
	return 0;
	} else {
		printf("differences %d / %d\n",sumdiff,  ucd1->uncompressedsize );
	return -1;
	}
}

struct TestData *loadTestData(char filename[], char name[]) {
	struct TestData *res;
	res = (struct TestData *) malloc(sizeof(struct TestData));
	res->name = strdup(name);
	res->ucd = loadTextFile(filename);
	char huffname[200];
		strcpy(huffname, filename);
		strcat(huffname, ".huff");
	res->cd = loadHuffFile(huffname);
	return res;
}

void freeTestData(struct TestData *td) {
	freeCompressedData(td->cd);
	freeUnCompressedData(td->ucd);
	free(td->name);
	free(td);
}

int tableHeight(struct HuffNode *tree, int r) {
	if (tree[r].izero == -1) {
		return 0;
	} else {
		return 1 + max(tableHeight(tree, tree[r].izero) , tableHeight(tree, tree[r].ione) );
	}
}

int treeSize(struct HuffNode *tree, int r) {
	if (tree[r].izero == -1) {
			return 1;
		} else {
			return 1 + treeSize(tree, tree[r].izero) +  treeSize(tree, tree[r].ione) ;
		}
}



int tableNumGroupsToGo(struct HuffNode *tree, int down, int bits, int r) {
    if (tree[r].izero == -1 && down == 0) {
    	return 0;
    } else if (tree[r].izero == -1 ) {
    	return 0;
    } else if (down == 0) {
    	return 1 + tableNumGroupsToGo(tree, bits, bits, r);
    } else {
    	return tableNumGroupsToGo(tree, down-1, bits, tree[r].izero) +  tableNumGroupsToGo(tree, down-1, bits, tree[r].ione);
    }
}

int tableNumGroups(struct HuffNode *tree, int bits, int r) {
    return 1 + tableNumGroupsToGo(tree, bits, bits, r);
}

int telescopedplusroot(struct HuffNode *tree, int bits, int r) {
	if (bits == 0) return 0;
	if (tree[r].izero == -1) {
			return 0;
		} else {
			return 1 + telescopedplusroot(tree, bits-1, tree[r].izero) + telescopedplusroot(tree, bits-1, tree[r].ione);
		}
}

int telescoped(struct HuffNode *tree, int bits, int r) {
	return telescopedplusroot(tree,bits,r) - 1;
}


int tableMinDepth(struct HuffNode *tree, int r) {
	if (tree[r].izero == -1) {
		return 0;
	} else {
		return 1 + min(tableMinDepth(tree, tree[r].izero) , tableMinDepth(tree, tree[r].ione) );
	}
}

void showDataBits(struct CompressedData *cd) {

	int pos;
	for (pos = 0;pos< cd->bits; pos++) {
		printf("%s", (((cd->data[pos / 8] >> (pos % 8)) & 1) ? "1" : "0"));
	}
	printf("\n");

}


void showHuffTable(struct HuffNode *nodes, int nc) {
	int i;
	for (i=0;i<nc;i++) {
		if (nodes[i].izero == -1) {
		printf("%d   '%c'\n",i, nodes[i].sym);
		} else {
			printf("%d   %d   %d\n",i, nodes[i].izero , nodes[i].ione);
		}
	}
}

