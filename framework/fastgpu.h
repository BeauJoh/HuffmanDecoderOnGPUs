/*
 * fastgpu.h
 *
 *  Created on: 06/01/2016
 *      Author: ericm
 */

#ifndef FASTGPU_H_
#define FASTGPU_H_
#include"decodeUtil.h"


void fastgpuApproach(struct CompressedData *cd,
                     struct UnCompressedData *uncompressed,
                     void *paramdata);

#endif /* FASTGPU_H_ */
