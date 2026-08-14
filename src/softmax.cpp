#include "softmax.h"

#include <cmath>

Tensor Softmax::forward(const Tensor& input) const
{
    Tensor output(input.shape());

    float maxValue = input[0];

    for (size_t i = 1; i < input.size(); i++) {
        if (input[i] > maxValue) {
            maxValue = input[i];
        }
    }

    float sum = 0.0f;

    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::exp(input[i] - maxValue);
        sum += output[i];
    }

    for (size_t i = 0; i < output.size(); i++) {
        output[i] /= sum;
    }

    return output;
}