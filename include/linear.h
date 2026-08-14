#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class Linear {
private:
    Tensor weights_;
    Tensor bias_;

public:
    Linear(size_t inputSize, size_t outputSize);

    Tensor& weights();
    Tensor& bias();

    Tensor forward(const Tensor& input) const;
};

#endif