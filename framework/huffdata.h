/*
 * huffdata.h
 *
 *  Created on: 03/12/2015
 *      Author: ericm
 */

#ifndef HUFFDATA_H_
#define HUFFDATA_H_


struct HuffNode {
	unsigned char sym;
	int izero;
	int ione;
};


struct TestData {
	struct CompressedData *cd;
	struct UnCompressedData *ucd;
	char *name;
};


struct CompressedData {
	int bits;
	int nodes;
	int uncompressedsize;
	struct HuffNode *tree;
	unsigned char *data;
};

struct UnCompressedData {
	int uncompressedsize;
	unsigned char *data;
};
struct TestData *loadTestData(char filename[], char name[]);
void freeTestData(struct TestData *td);

struct UnCompressedData *loadTextFile(char filename[]);
struct UnCompressedData *newUnCompressedData(int size);
void freeUnCompressedData(struct UnCompressedData *cd);

void showDataBits(struct CompressedData *cd);
void showHuffTree(struct HuffNode *nodes, int root);
void showHuffTable(struct HuffNode *nodes, int nc);
void listHuffCodes(struct HuffNode *nodes, int root);

int compareUnCompressedData(struct UnCompressedData *ucd1, struct UnCompressedData *ucd2);
void clearUnCompressedData(struct UnCompressedData *cd);

struct CompressedData *loadHuffFile(char filename[]);
void freeCompressedData(struct CompressedData *cd);
void infoCompressedData(struct CompressedData *cd);
int telescoped(struct HuffNode *tree, int bits, int r);
int tableHeight(struct HuffNode *tree, int r);
int tableMinDepth(struct HuffNode *tree, int r);
int treeSize(struct HuffNode *tree, int r);
int tableNumGroups(struct HuffNode *tree, int bits, int r);

void infoTestData(struct TestData *td);

void showbits2(int len, int bits);

#endif /* HUFFDATA_H_ */
