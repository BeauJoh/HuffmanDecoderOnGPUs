#define KERNELSTART unsigned int idx = get_group_id(0) * work_per_thread * stride + get_local_id(0); unsigned int bit = idx; for (unsigned int i = 0; i < work_per_thread; i++) { if(bit < bitsincomp){
#define KERNELEND } bit += stride; }

__kernel void initbitsindex(__global int* bitsindex_d,
                            const int bitsincomp,
                            const int work_per_thread,
                            const int stride)
{
    KERNELSTART
			bitsindex_d[bit] = -1;
    KERNELEND
}

