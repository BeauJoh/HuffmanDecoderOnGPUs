/*
 * fastgpu.cu
 *
 *  Created on: 06/01/2016
 *      Author: ericm
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "decodeUtil.h"
#include "huffdata.h"

extern void checkMalloc(cudaError_t retval, const char *message);

struct bigTableSimpleOp1 {
	unsigned char bitsused;
	unsigned char sym;
};

__global__ void lookupsymbolsSimpleOpt1(int tablesize,  int work_per_thread, int stride, struct HuffNode *tree,
		struct bigTableSimpleOp1 *te, int h) {


	int idx = (blockIdx.x * work_per_thread * stride + threadIdx.x);
	int i, tableindex; tableindex = idx;
	for (i = 0; i < work_per_thread; i++) {	if (tableindex < tablesize) {

	int pos = 0;
	int nextbit;
	int tablepos = 0;
	int notfound = 1;
	while (pos < h && notfound) {
		nextbit = (tableindex >> pos) & 1;
		tablepos = (nextbit ? tree[tablepos].ione : tree[tablepos].izero);
		if (tree[tablepos].ione == -1) { // its a leaf
			te[tableindex].sym = tree[tablepos].sym;
			te[tableindex].bitsused = pos + 1;
			notfound = 0;
		}
		pos++;
	}


	}
	tableindex += stride;
	}
}




int tableMinDepthOpt(struct HuffNode *tree, int r) {
	if (tree[r].izero == -1) {
		return 0;
	} else {
		return 1 + min(tableMinDepthOpt(tree, tree[r].izero) , tableMinDepthOpt(tree, tree[r].ione) );
	}
}

#define KERNELSTART int idx = (blockIdx.x * work_per_thread * stride + threadIdx.x); int i, bit; bit = idx; for (i = 0; i < work_per_thread; i++) {	if (bit < bitsincomp) {
#define KERNELEND } bit += stride; }

// Kernel that executes on the GPU init the bitsindex
__global__ void initbitsindexOpt1(int *bitsindex_d, int bitsincomp,
		int work_per_thread, int stride) {

	    KERNELSTART
			bitsindex_d[bit] = -1;
		KERNELEND

}


__global__ void decodeAllBitsOpt1(int bitsincomp, int work_per_thread, int stride,
		struct HuffNode *table_d, unsigned char *data_d, unsigned char *bitdecode_d,
		int *bitsteps_d) {
	int pos, tablepos, nextbit;

	KERNELSTART

			pos = bit;
			tablepos = 0;
			while ((table_d[tablepos].izero != -1) && pos < bitsincomp) {
				nextbit = (data_d[pos / 8] >> (pos % 8)) & 1;
				tablepos =
						(nextbit ?
								table_d[tablepos].ione : table_d[tablepos].izero);
				pos++;
			}

			bitdecode_d[bit] = table_d[tablepos].sym;
			bitsteps_d[bit] = pos - bit;

		KERNELEND
}


__global__ void makebigtableOpt1(int bitsincomp, int work_per_thread, int stride,
		struct HuffNode *table_d, unsigned char *data_d, unsigned char *bitdecode_d,
		int *bitstepsA_d,int bitstepsAsize, int *bitstepsB_d, int bitstepsBsize, int step, int powertwo, int *bitsteoresult_d) {
	int s, w;
	KERNELSTART

			s = (bit>=bitstepsAsize ? -1 :  bitstepsA_d[bit]);
			if (s == -1 || bit + s > bitsincomp) {
				if (bit < bitstepsBsize) bitstepsB_d[ bit] = -1;
			} else {
				w = (bit+s>=bitstepsAsize ? -1 : bitstepsA_d[ bit + s]);
				if (w == -1 || bit + s + w > bitsincomp) {
					if (bit < bitstepsBsize) bitstepsB_d[ bit] = -1;
				} else {
					if (bit < bitstepsBsize) bitstepsB_d[ bit] = s + w;
				}
			}

			//bitsteps_d[bitsincomp * (step + 1) + bit] = ((s == -1 || bit + s > bitsincomp) ? -1  : (((w= bitsteps_d[bitsincomp * step + bit + s]) == -1 || bit + s + w > bitsincomp)?-1:s+w));

		KERNELEND
   if (idx == 0)
	*bitsteoresult_d = bitstepsA_d[0];
}


__global__ void calcbitsindexOpt1(int bitsincomp,
                              int work_per_thread,
                              int stride,
                              int *bitsindex_d,
                              int *bitstepsA_d,
							  int bitstepsAsize,
							  int *bitstepsM_d,
							  int bitstepsMsize,
                              int step,
                              int powertwo) {
    KERNELSTART

			int offset = (bit >= bitstepsMsize? -1 : bitstepsM_d[ bit]);
			int curval = bitsindex_d[bit];
			if (offset != -1 && curval != -1 && bit + offset < bitsincomp) {
				bitsindex_d[bit + offset] = curval + powertwo;

			}

    KERNELEND
}


__global__ void calcresultOpt1(int bitsincomp,
                           int work_per_thread,
                           int stride,
                           int *bitsindex_d,
                           unsigned char *bitdecode_d,
                           unsigned char *result_d) {
	KERNELSTART


			if (bitsindex_d[bit] != -1) {
				result_d[bitsindex_d[bit]] = bitdecode_d[bit];
			}

    KERNELEND
}

__global__ void findmaxOpt1(int bitsincomp, int *bitsindex_d) {
	int bit = bitsincomp - 1;
	while (bit > 0 && bitsindex_d[bit] == -1)
		bit--;
	bitsindex_d[0] = bitsindex_d[bit];

}


extern "C" void fastgpuApproachOpt1(struct CompressedData *cd,
                                struct UnCompressedData *uncompressed,
                                void *paramdata) {

    int resultsize;
    int bitsincomp = cd->bits;
	int block_size = 200;//1024 or 124??;

	int work_per_thread = 400;//1100 or 100??;
	int stride = block_size;
	int n_blocks = (bitsincomp / (block_size * work_per_thread)) +
        ((bitsincomp % (block_size * work_per_thread)) == 0 ? 0 : 1) ;
#ifdef FGPUDEBUG
	printf("work_per_thread : %d n_blocks : %d bitsincomop : %d\n",
			work_per_thread, n_blocks, bitsincomp);
#endif
	int *bitsindex_d;

	unsigned char* data_d;

	int** bitsteps_d;
	int* bitstepssize_h;
	struct HuffNode *table_d;
	cudaStream_t stream1, stream2;

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	unsigned char *bitdecode_d, *result_d;

	int tablemindepth = tableMinDepthOpt(cd->tree,0);

	//printf("%d\n",tablemindepth);

	checkMalloc(cudaMalloc((void **) &bitsindex_d, bitsincomp * sizeof(int)),
			"bitsindex_d");

	initbitsindexOpt1 <<< n_blocks, block_size, 0 ,stream1 >>> (bitsindex_d, bitsincomp, work_per_thread, stride);
	int zerovalue = 0;
	cudaMemcpyAsync(bitsindex_d, &zerovalue, sizeof(int), cudaMemcpyHostToDevice,stream1);


    // set up : table, data, bitdecode, bitsteps
	checkMalloc(
			cudaMalloc((void **) &table_d,  cd->nodes* sizeof(struct HuffNode)),
			"table_d");
	 cudaMemcpyAsync(table_d, cd->tree, cd->nodes * sizeof(struct HuffNode),cudaMemcpyHostToDevice,stream2);

	int datasize = (
			(bitsincomp % 8) == 0 ? bitsincomp / 8 : (bitsincomp / 8) + 1);


	checkMalloc(cudaMalloc((void **) &data_d, datasize), "data_d");
	cudaMemcpyAsync(data_d, cd->data, datasize, cudaMemcpyHostToDevice, stream2);


	bitsteps_d = (int **) malloc(25*sizeof(int *));
	bitstepssize_h = (int *) malloc(25*sizeof(int));

	checkMalloc(
				cudaMalloc((void **) &(bitsteps_d[0]), bitsincomp * sizeof(int)),   // maybe 24 ???
				"bitsteps_d");
    bitstepssize_h[0] = bitsincomp;

    checkMalloc(cudaMalloc((void **) &bitdecode_d, bitsincomp * sizeof(unsigned char)),
    			"bitdecode_d");

    decodeAllBitsOpt1 <<< n_blocks, block_size,0,stream2 >>> (bitsincomp, work_per_thread, stride, table_d, data_d, bitdecode_d, bitsteps_d[0]);


    int aless = tablemindepth;
    int asize;
	for (int i=1;i<25;i++) {
		asize = bitsincomp - aless;
			        if (asize < 0) asize = 0;
	checkMalloc(

			cudaMalloc((void **) &(bitsteps_d[i]), asize * sizeof(int)),   // maybe 24 ???
			"bitsteps_d");
	        bitstepssize_h[i] = asize;
	        aless = 2 * aless;
	}

	checkMalloc(cudaMalloc((void **) &result_d, bitsincomp * sizeof(int)),
			"result_d");




	int step = 0;
	int powertwo = 1;

	int bitsteoresult;
	int *bitsteoresult_d;

	checkMalloc(cudaMalloc((void **) &bitsteoresult_d, sizeof(int)),
			"bitsteoresult_d");

#ifdef FGPUDEBUG
    printf("Printing a sample of bitdecode_h\n");
    int i,j;
    unsigned char *bitdecode_h;
    bitdecode_h = (unsigned char *) malloc(bitsincomp * sizeof(unsigned char));
	cudaMemcpy(bitdecode_h, bitdecode_d, bitsincomp * sizeof(unsigned char),
				cudaMemcpyDeviceToHost);
	for (i=0;i<15;i++) {
		if (bitdecode_h[i] == '\n') printf("   \\n "); else
		printf("    %c ",bitdecode_h[i]);
	}
    printf("\n");


    int *bitsteps_h;
       bitsteps_h = (int *) malloc(bitsincomp * 25 * sizeof(int));
#endif
//exit(-1);

    cudaStreamSynchronize(stream2);

	do {
#ifdef FGPUDEBUG
		printf("Make Step Tree step %d %d\n", step, bitsteoresult);
#endif


		makebigtableOpt1 <<< n_blocks, block_size >>> (bitsincomp, work_per_thread, stride, table_d, data_d, bitdecode_d, bitsteps_d[step],bitstepssize_h[step],bitsteps_d[step+1],bitstepssize_h[step+1], step, powertwo, bitsteoresult_d);

		cudaMemcpy(&bitsteoresult, bitsteoresult_d, sizeof(int),
				cudaMemcpyDeviceToHost);

		step++;
		powertwo = powertwo << 1;
	} while (bitsteoresult != -1);

#ifdef FGPUDEBUG
 	cudaMemcpy(bitsteps_h, bitsteps_d, bitsincomp * 25 * sizeof(int),
   				cudaMemcpyDeviceToHost);
   	for (j=0;j<5;j++) {
   	for (i=0;i<15;i++) {

   		printf("  %3d ",bitsteps_h[j*bitsincomp + i]);
   	}
       printf("\n");
   	}
#endif



    cudaStreamSynchronize(stream1);
	powertwo = powertwo >> 1;

#ifdef FGPUDEBUG
	printf("calcbits index\n");
#endif

	while (step > 0) {
		calcbitsindexOpt1 <<< n_blocks, block_size >>> (bitsincomp, work_per_thread, stride, bitsindex_d, bitsteps_d[step], bitstepssize_h[step], bitsteps_d[step-1], bitstepssize_h[step-1], step, powertwo);
		step--;
		powertwo = powertwo >> 1;
	}

#ifdef FGPUDEBUG
	printf("calc result\n");
#endif

	//work out result
	calcresultOpt1 <<< n_blocks, block_size >>> (bitsincomp, work_per_thread, stride, bitsindex_d, bitdecode_d, result_d );

#ifdef FGPUDEBUG
	printf("find max\n");
#endif


	int maxvalue;
	findmaxOpt1<<<1,1>>>(bitsincomp, bitsindex_d);

	cudaMemcpy(&maxvalue, bitsindex_d, sizeof(int), cudaMemcpyDeviceToHost);

	// obtain result from GPU
	resultsize = maxvalue + 1;
	cudaMemcpy(uncompressed->data, result_d, resultsize, cudaMemcpyDeviceToHost);

	cudaFree(bitsindex_d);
	cudaFree(table_d);
	cudaFree(data_d);

	for (int i=0;i<25;i++) {
	    cudaFree(bitsteps_d[i]);
	}
	free(bitsteps_d);
	free(bitstepssize_h);


	cudaFree(bitdecode_d);
	cudaFree(result_d);
	cudaFree(bitsteoresult_d);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

}

/*
extern "C" void fastgpuApproach(struct CompressedData *cd,
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
	decodeHuffFastGPU<<<1,1>>>(uncompressed->uncompressedsize, uncompresseddata_d, cd->bits, table_d, compresseddata_d);
	cudaMemcpy(uncompressed->data, uncompresseddata_d,
			uncompressed->uncompressedsize, cudaMemcpyDeviceToHost);

} */
