#include "linear.h"

Linear::Linear(size_t inputSize, size_t outputSize)
    : weights_({inputSize, outputSize}),
      bias_({1, outputSize})
{
}

Tensor& Linear::weights()
{
    return weights_;
}

Tensor& Linear::bias()
{
    return bias_;
}

Tensor Linear::forward(const Tensor& input) const //output = input x weight + bias calc
{
    Tensor output = input.matmul(weights_);

    size_t rows = output.shape()[0];
    size_t cols = output.shape()[1];

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            output.at(row, col) += bias_.at(0, col);
        }
    }

    return output;
}