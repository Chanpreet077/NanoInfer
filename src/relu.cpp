#include "relu.h"

Tensor ReLU::forward(const Tensor& input) const
{
    Tensor output(input.shape());

    for (size_t i = 0; i < input.size(); i++) {
        if (input[i] > 0.0f) {
            output[i] = input[i];
        } else {
            output[i] = 0.0f;
        }
    }

    return output;
}