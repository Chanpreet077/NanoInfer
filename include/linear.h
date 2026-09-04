#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include "quantization.h"

class Linear {
private:
    Tensor weights_;
    Tensor bias_;
    QuantizedTensor quantizedWeights_;

public:
    Linear(size_t inputSize, size_t outputSize);

    Tensor& weights();
    Tensor& bias();

    void quantizeWeights();

    Tensor forward(
        const Tensor& input,
        bool useCuda = false,
        bool useInt8 = false
    ) const;
};

#endif