struct HuffNode {
	unsigned char sym;
	int izero;
	int ione;
};

#define KERNELSTART unsigned int idx = get_group_id(0) * work_per_thread * stride + get_local_id(0); unsigned int bit = idx; for (unsigned int i = 0; i < work_per_thread; i++) { if(bit < bitsincomp){
#define KERNELEND } bit += stride; }

__kernel void decodeallbits(const int bitsincomp,
                            const int work_per_thread,
                            const int stride,
	                        __global struct HuffNode *table_d,
                            __global unsigned char *data_d,
                            __global unsigned char *bitdecode_d,
                            __global int *bitsteps_d)
{
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

