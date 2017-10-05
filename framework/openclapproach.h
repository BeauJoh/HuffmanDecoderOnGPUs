/*
 * fastgpu.h
 *
 *  Created on: 06/01/2016
 *      Author: ericm
 */
#pragma once
#include"huffdata.h"
#include"decodeUtil.h"

void openclApproach(struct CompressedData *cd,
                     struct UnCompressedData *uncompressed,
                     void *paramdata);

