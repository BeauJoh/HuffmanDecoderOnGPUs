# HuffmanDecoderOnGPUs

Huffman encoding provides a simple approach for lossless compression of
sequential data. The length of encoded symbols varies and these symbols are
tightly packed in the compressed data. Thus Huffman decoding is not easily
parallelisable. This is unfortunate since it is desirable to have a parallel
algorithm which scales with the increased core count of modern systems. This
paper presents a parallel approach for decoding Huffman codes which work by
decoding from every location in the bit sequence then concurrently combining
the results into the uncompressed sequence. Although requiring more operations
then serial approaches the presented approach is able to produce results
marginally faster, on sufficiently large data sets, then that of a simple
serial implementation. This is achieved by using the large number of threads
available on modern GPUs. A variety of implementations, primarily OpenCL, are
presented to demonstrate the scaling of this algorithm on CPU and GPU hardware
in response to cores available. As devices with more cores become available the
importance of such an algorithm will increase.

The associated software in this repository is an examination of Huffman
Decoding using parallel methods on GPU and other accelerator architectures, and
is implemented with OpenCL, CUDA and C -- there is also an experimental OpenACC
version.


Compilation
-----------
There are suitable `Makefiles` in each software release. Therefore, `cd` to the
`framework` directory and either `ReleaseCL` for the OpenCL version,
`ReleaseCU` for the CUDA implementation or `ReleaseACC` for the OpenACC version
-- though this is highly unstable. Then run `make`. These `Makefiles` were
generated on one particular machine and with fixed dependency code paths, thus
some modification may be needed to compile on your system.


Datasets
--------
The `files` directory contains the data sets used in our experimental analysis.
Uncompressed files included are the `E.coli`, `bible.txt`, `book2`, `hello`,
`kjv.txt`, `news`, `paper1` and `world192.txt` datasets, these are added to be
used for legitimacy of solution checks.
Such that, the decoded solution of the encoded files should be the same as the
uncompressed files presented here.  The compressed files are therefore all the
corresponding files with a `.huff` extension.


Usage
-----



License Terms (MIT)
-------------------
Copyright (c) 2017 Beau Johnston <beau.johnston@anu.edu.au> and Eric C.
McCreath <eric.mccreath@anu.edu.au>, The Australian National University.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

If you publish using this software or a modified version of it, we would
appreciate your citing the following paper:

    Beau Johnston and Eric C. McCreath. Parallel Huffman Decoding: Presenting a
    Fast and Scalable Algorithm for Increasingly Multicore Devices.
    The 15th IEEE International Symposium on Parallel and Distributed
    Processing with Applications (ISPA), December, 2017.

