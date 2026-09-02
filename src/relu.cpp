#include "relu.h"

Tensor ReLU::forward(const Tensor& input) const //after going thru linear (784-128) we go thru this RELU
{
    Tensor output(input.shape());

    for (size_t i = 0; i < input.size(); i++) { //we turn neg inputs to 0 to make things nonlinear/more expressive (lets signals activate or become 0)
        if (input[i] > 0.0f) {
            output[i] = input[i];
        } else {
            output[i] = 0.0f;
        }
    }

    return output;
}