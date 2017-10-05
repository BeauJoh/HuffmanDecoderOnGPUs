
__kernel void findmax(const int bitsincomp,
                      __global int *bitsindex_d) {
	int bit = bitsincomp - 1;
	while (bit > 0 && bitsindex_d[bit] == -1)
		bit--;
	bitsindex_d[0] = bitsindex_d[bit];
}

