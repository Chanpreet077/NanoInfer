#ifndef RELU_H
#define RELU_H

#include "tensor.h"

class ReLU {
public:
    Tensor forward(const Tensor& input) const;
};

#endif