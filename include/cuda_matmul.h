#ifndef CUDA_MATMUL_H
#define CUDA_MATMUL_H

#include "tensor.h"

Tensor cudaMatmul(const Tensor& a, const Tensor& b);

#endif