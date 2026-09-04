#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include "tensor.h"

#include <cstdint>
#include <vector>

struct QuantizedTensor
{
    std::vector<int8_t> data;
    std::vector<size_t> shape;
    float scale;
};

QuantizedTensor quantizeTensor(const Tensor& tensor);

#endif