
#define KERNELSTART unsigned int idx = get_group_id(0) * work_per_thread * stride + get_local_id(0); unsigned int bit = idx; for (unsigned int i = 0; i < work_per_thread; i++) { if(bit < bitsincomp){
#define KERNELEND } bit += stride; }

__kernel void calcresult(const int bitsincomp,
                         const int work_per_thread,
                         const int stride,
                         __global int *bitsindex_d,
                         __global unsigned char *bitdecode_d,
                         __global unsigned char *result_d) {
	KERNELSTART


			if (bitsindex_d[bit] != -1) {
				result_d[bitsindex_d[bit]] = bitdecode_d[bit];
			}

    KERNELEND
}

