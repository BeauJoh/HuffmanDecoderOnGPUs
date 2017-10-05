
#define KERNELSTART unsigned int idx = get_group_id(0) * work_per_thread * stride + get_local_id(0); unsigned int bit = idx; for (unsigned int i = 0; i < work_per_thread; i++) { if(bit < bitsincomp){
#define KERNELEND } bit += stride; }

__kernel void calcbitsindex(const int bitsincomp,
                            const int work_per_thread,
                            const int stride,
                            __global int *bitsindex_d,
                            __global int *bitsteps_d,
                            const int step,
                            const int powertwo) {
    KERNELSTART

			int offset = bitsteps_d[bitsincomp * (step - 1) + bit];
			int curval = bitsindex_d[bit];
			if (offset != -1 && curval != -1 && bit + offset < bitsincomp) {
				bitsindex_d[bit + offset] = curval + powertwo;

			}

    KERNELEND
}

