/*
 * decodeUtil.h
 *
 *  Created on: 03/12/2015
 *      Author: ericm
 */

#ifndef DECODEUTIL_H_
#define DECODEUTIL_H_

#include"huffdata.h"


struct decoder {
	void (*decoder_function)(struct CompressedData *cd,
			struct UnCompressedData *uncompressed, void *paramdata);
	void *paramdata;
	char *name;
};

struct decoder *newDecoder(
		void (*decoder_function),
		void *paramdata, char *name);
void freeDecoder(struct decoder *d);

#define REPEATS 25

#define max(A,B) ((A)<(B)?(B):(A))
#define min(A,B) ((A)<(B)?(A):(B))

double evaluate(struct decoder *d, struct TestData *td, int withcheck);


#endif /* DECODEUTIL_H_ */
