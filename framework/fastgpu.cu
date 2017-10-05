/*
 * fastgpu.cu
 *
 *  Created on: 06/01/2016
 *      Author: ericm
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "decodeUtil.h"
extern "C"{
#include "timing.h"
}
void checkMalloc(cudaError_t retval, const char *message) {

	if (retval != cudaSuccess) {
		printf("cudaMalloc error: %s %s\n", message,
				cudaGetErrorString(retval));
		exit(-1);
	}
}

void checkGPU(const char* str){
	cudaError_t err = cudaPeekAtLastError();
	if (err != cudaSuccess){
		printf("Cuda failed with %s in %s\n",cudaGetErrorString(err),str);
		exit(-1);
	}
}

#define KERNELSTART int idx = (blockIdx.x * work_per_thread * stride + threadIdx.x); int i, bit; bit = idx; for (i = 0; i < work_per_thread; i++) {	if (bit < bitsincomp) {
#define KERNELEND } bit += stride; }

// Kernel that executes on the GPU init the bitsindex
__global__ void initbitsindex(int *bitsindex_d, int bitsincomp,
		int work_per_thread, int stride) {

	    KERNELSTART
			bitsindex_d[bit] = -1;
	    KERNELEND
}


__global__ void decodeAllBits(int bitsincomp, int work_per_thread, int stride,
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
//bitdecode_d[bit] = 'a';
			bitdecode_d[bit] = table_d[tablepos].sym;
			bitsteps_d[bit] = pos - bit;

	KERNELEND
}


__global__ void makebigtable(int bitsincomp, int work_per_thread, int stride,
		struct HuffNode *table_d, unsigned char *data_d, unsigned char *bitdecode_d,
		int *bitsteps_d, int step, int powertwo, int *bitsteoresult_d) {
	int s, w;
	KERNELSTART

			s = bitsteps_d[bitsincomp * step + bit];
			if (s == -1 || bit + s > bitsincomp) {
				bitsteps_d[bitsincomp * (step + 1) + bit] = -1;
			} else {
				w = bitsteps_d[bitsincomp * step + bit + s];
				if (w == -1 || bit + s + w > bitsincomp) {
					bitsteps_d[bitsincomp * (step + 1) + bit] = -1;
				} else {
					bitsteps_d[bitsincomp * (step + 1) + bit] = s + w;
				}
			}

			//bitsteps_d[bitsincomp * (step + 1) + bit] = ((s == -1 || bit + s > bitsincomp) ? -1  : (((w= bitsteps_d[bitsincomp * step + bit + s]) == -1 || bit + s + w > bitsincomp)?-1:s+w));

	KERNELEND

   if (idx == 0)
	*bitsteoresult_d = bitsteps_d[bitsincomp * step + 0];
}


__global__ void calcbitsindex(int bitsincomp,
                              int work_per_thread,
                              int stride,
                              int *bitsindex_d,
                              int *bitsteps_d,
                              int step,
                              int powertwo) {
    KERNELSTART

			int offset = bitsteps_d[bitsincomp * (step - 1) + bit];
			int curval = bitsindex_d[bit];
			if (offset != -1 && curval != -1 && bit + offset < bitsincomp) {
				bitsindex_d[bit + offset] = curval + powertwo;

			}

    KERNELEND
}


__global__ void calcresult(int bitsincomp,
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

__global__ void findmax(int bitsincomp, int *bitsindex_d) {
	int bit = bitsincomp - 1;
	while (bit > 0 && bitsindex_d[bit] == -1)
		bit--;
	bitsindex_d[0] = bitsindex_d[bit];

}

extern "C" void fastgpuApproach(struct CompressedData *cd,
                                struct UnCompressedData *uncompressed,
                                void *paramdata) {
	//struct timer *t;
	//t = newTimer();
	int block_size = 1024;
	int work_per_thread = 1;
	//for (int work_per_thread = 1; work_per_thread <= 1024; work_per_thread++){
	//double samples = 0;
	//double mean = 0.0;
	//double variance = 0.0;
	//for (int repeats = 100; repeats > 0; repeats --){
	//timestart(t);
	int resultsize;
	int bitsincomp = cd->bits;

	//int block_size = 200;
	//int work_per_thread = 64;//400;

	int stride = block_size;
	int n_blocks = (bitsincomp / (block_size * work_per_thread)) +
        ((bitsincomp % (block_size * work_per_thread)) == 0 ? 0 : 1) ;

	assert(work_per_thread > bitsincomp/(block_size*n_blocks));
	/*
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	printf("name = %s\n",prop.name);
	printf("warpsize = %i\n",prop.warpSize);
	printf("clockrate = %f GHz\n",prop.clockRate*1e-6);
	printf("multiprocessor count = %i\n",prop.multiProcessorCount);
	printf("max threads per multiprocessor = %i\n",prop.maxThreadsPerMultiProcessor);
	printf("bitsincomp = %i\t",bitsincomp);
	printf("block_size = %i\t",block_size);
	printf("work_per_thread = %i\t",work_per_thread);
	printf("n_blocks = %i\n",n_blocks);
	*/
	assert(n_blocks < 65535 && "65535 blocks per diamension of the grid, given only 1 dimension is used in this application");

#ifdef FGPUDEBUG
	printf("work_per_thread : %d n_blocks : %d bitsincomop : %d\n",
			work_per_thread, n_blocks, bitsincomp);
#endif
	int *bitsindex_d;
	struct HuffNode *table_d;
	unsigned char* data_d;
	int *bitsteps_d;
	unsigned char *bitdecode_d, *result_d;

	checkMalloc(cudaMalloc((void **) &bitsindex_d, bitsincomp * sizeof(int)),
			"bitsindex_d");
	
	initbitsindex <<< n_blocks, block_size >>> (bitsindex_d, bitsincomp, work_per_thread, stride);
	checkGPU("initbitsindex");

	// set up : table, data, bitdecode, bitsteps
	checkMalloc(cudaMalloc((void **) &table_d,  cd->nodes* sizeof(struct HuffNode)),"table_d");
	cudaMemcpy(table_d, cd->tree, cd->nodes * sizeof(struct HuffNode),cudaMemcpyHostToDevice);

	int datasize = ((bitsincomp % 8) == 0 ? bitsincomp / 8 : (bitsincomp / 8) + 1);
	checkMalloc(cudaMalloc((void **) &data_d, datasize), "data_d");
	cudaMemcpy(data_d, cd->data, datasize, cudaMemcpyHostToDevice);

	checkMalloc(cudaMalloc((void **) &bitsteps_d, 25 * bitsincomp * sizeof(int)),   // maybe 24 ???
		    "bitsteps_d");

	checkMalloc(cudaMalloc((void **) &result_d, bitsincomp * sizeof(int)),
			"result_d");

	checkMalloc(cudaMalloc((void **) &bitdecode_d, bitsincomp * sizeof(unsigned char)),
			"bitdecode_d");
#ifdef FGPUDEBUG
	printf("Memory Allocated\n");
#endif
	decodeAllBits <<< n_blocks, block_size >>> (bitsincomp, work_per_thread, stride, table_d, data_d, bitdecode_d, bitsteps_d);
	checkGPU("decodeAllBits");

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

	do {
#ifdef FGPUDEBUG
		printf("Make Step Tree step %d %d\n", step, bitsteoresult);
#endif

		
		makebigtable <<< n_blocks, block_size >>> (bitsincomp, work_per_thread, stride, table_d, data_d, bitdecode_d, bitsteps_d, step, powertwo, bitsteoresult_d);
		checkGPU("makebigtable");

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

	int zerovalue = 0;
	powertwo = powertwo >> 1;
	cudaMemcpy(bitsindex_d, &zerovalue, sizeof(int), cudaMemcpyHostToDevice);

#ifdef FGPUDEBUG
	printf("calcbits index\n");
#endif

	while (step > 0) {
		calcbitsindex <<< n_blocks, block_size >>> (bitsincomp, work_per_thread, stride, bitsindex_d, bitsteps_d, step, powertwo);
		checkGPU("calcbitsindex");
		step--;
		powertwo = powertwo >> 1;
	}

#ifdef FGPUDEBUG
	printf("calc result\n");
#endif

	//work out result
	calcresult <<< n_blocks, block_size >>> (bitsincomp, work_per_thread, stride, bitsindex_d, bitdecode_d, result_d );
	checkGPU("calcresult");

#ifdef FGPUDEBUG
	printf("find max\n");
#endif


	int maxvalue;
	findmax<<<1,1>>>(bitsincomp, bitsindex_d);
	checkGPU("findmax");

	cudaMemcpy(&maxvalue, bitsindex_d, sizeof(int), cudaMemcpyDeviceToHost);

	// obtain result from GPU
	resultsize = maxvalue + 1;
	cudaMemcpy(uncompressed->data, result_d, resultsize, cudaMemcpyDeviceToHost);

	cudaFree(bitsindex_d);
	cudaFree(table_d);
	cudaFree(data_d);
	cudaFree(bitsteps_d);
	cudaFree(bitdecode_d);
	cudaFree(result_d);
	cudaFree(bitsteoresult_d);
	//timestop(t);
	//determine mean and variance using Welford's Algorithm:
	//if(repeats != 100){//the first run has some startup cost
	//++samples;
	//double new_mean = mean + (timerms(t)-mean)/samples;
	//variance += (timerms(t) - mean) * (timerms(t)-new_mean);
	//mean = new_mean;
	//}
	//printf("work_per_thread %i took %g +- %g ms\n\n",work_per_thread,mean,variance/samples);
	//printf("blocksize %i and work_per_thread %i took %g ms\n\n",blocksize,work_per_thread,timerms(t));
	//printf("(%.*s)",1024,uncompressed->data+resultsize-1024);
	//}
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
