#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "tensor.h"

class Softmax {
public:
    Tensor forward(const Tensor& input) const;
};

#endif