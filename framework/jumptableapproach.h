/*
 * jumptableapproach.h
 *
 *  Created on: 08/12/2015
 *      Author: ericm
 */

#ifndef JUMPTABLEAPPROACH_H_
#define JUMPTABLEAPPROACH_H_

#include"huffdata.h"
#include"decodeUtil.h"

struct tableIndex {
	int prebitsnum;
	int prebits;
};

struct jumpElement3 {
	int nextTable;
	unsigned char numSym;
	unsigned char syms[3];
};

struct jumpElement7 {
	int nextTable;
	unsigned char numSym;
	unsigned char syms[7];
};



void jumptableApproach(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata);

#endif /* JUMPTABLEAPPROACH_H_ */
