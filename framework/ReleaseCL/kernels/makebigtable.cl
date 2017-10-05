struct HuffNode {
	unsigned char sym;
	int izero;
	int ione;
};

#define KERNELSTART unsigned int idx = get_group_id(0) * work_per_thread * stride + get_local_id(0); unsigned int bit = idx; for (unsigned int i = 0; i < work_per_thread; i++) { if(bit < bitsincomp){
#define KERNELEND } bit += stride; }

__kernel void makebigtable(const int bitsincomp,
                           const int work_per_thread,
                           const int stride,
                           __global struct HuffNode *table_d,
                           __global unsigned char *data_d,
                           __global unsigned char *bitdecode_d,
                           __global int *bitsteps_d,
                           const int step,
                           const int powertwo,
                           __global int *bitsteoresult_d) {
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

    KERNELEND

   if (idx == 0) {
       *bitsteoresult_d = bitsteps_d[bitsincomp * step + 0];
   }
}

