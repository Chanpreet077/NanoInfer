#include "softmax.h" //turns scores -> probabilities

#include <cmath>

Tensor Softmax::forward(const Tensor& input) const
{
    Tensor output(input.shape()); //output must have same shape as input

    float maxValue = input[0];

    for (size_t i = 1; i < input.size(); i++) { //find maxVal
        if (input[i] > maxValue) {
            maxValue = input[i];
        }
    }

    float sum = 0.0f;

    for (size_t i = 0; i < input.size(); i++) {  //do e^(score - maxVal) for each score
        output[i] = std::exp(input[i] - maxValue);
        sum += output[i]; //get sum of new calculated scored after exponentating
    }

    for (size_t i = 0; i < output.size(); i++) { //divide each output by total output sum to get probabilities
        output[i] /= sum;
    }

    return output;
}