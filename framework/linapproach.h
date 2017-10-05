/*
 * linapproach.h
 *
 *  Created on: 22/12/2015
 *      Author: ericm
 */

#ifndef LINAPPROACH_H_
#define LINAPPROACH_H_
#include"huffdata.h"
#include"decodeUtil.h"

struct sElement8 {
	union {
		int val;
		unsigned char syms[8];
	};
	int l;  // this is the L table in lin.
	unsigned char numSym;

};

void linApproach(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata);

#endif /* LINAPPROACH_H_ */
