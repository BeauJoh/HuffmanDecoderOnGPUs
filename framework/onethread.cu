/*
 * onethread.cu
 *
 *  Created on: 06/01/2016
 *      Author: ericm
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "decodeUtil.h"


__global__ void decodeHuffOneThread(int uncompsize,
		unsigned char *uncompresseddata_d, int bits, struct HuffNode *table_d,
		unsigned char *compresseddata_d) {
	int pos = 0;
	int nextbit;
	int tablepos = 0;
	int resultpos = 0;
	while (pos < bits) {
		nextbit = (compresseddata_d[pos / 8] >> (pos % 8)) & 1;
		tablepos = (nextbit ? table_d[tablepos].ione : table_d[tablepos].izero);
		if (table_d[tablepos].ione == -1 && table_d[tablepos].izero == -1) { // its a leaf
			uncompresseddata_d[resultpos] = table_d[tablepos].sym;
			resultpos++;
			tablepos = 0;
		}
		pos++;
	}

}

extern "C" void onethreadApproach(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {

	unsigned char *uncompresseddata_d;
	struct HuffNode *table_d;
	unsigned char *compresseddata_d;

	int compbytes = cd->bits / 8 + (cd->bits % 8 == 0 ? 0 : 1);

	cudaMalloc(&uncompresseddata_d, uncompressed->uncompressedsize);
	cudaMalloc(&table_d, cd->nodes * sizeof(struct HuffNode));
	cudaMalloc(&compresseddata_d, compbytes);
	cudaMemcpy(table_d, cd->tree, cd->nodes * sizeof(struct HuffNode),
			cudaMemcpyHostToDevice);
	cudaMemcpy(compresseddata_d, cd->data, compbytes, cudaMemcpyHostToDevice);
	decodeHuffOneThread<<<1,1>>>(uncompressed->uncompressedsize, uncompresseddata_d, cd->bits, table_d, compresseddata_d);
	cudaMemcpy(uncompressed->data, uncompresseddata_d,
			uncompressed->uncompressedsize, cudaMemcpyDeviceToHost);

}
