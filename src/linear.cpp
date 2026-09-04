#include "linear.h"
#include "cuda_matmul.h"

#include <stdexcept>

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

void Linear::quantizeWeights()
{
    quantizedWeights_ = quantizeTensor(weights_);
}

Tensor Linear::forward(
    const Tensor& input,
    bool useCuda,
    bool useInt8
) const //output = input x weight + bias calc
{
    if (useInt8)
    {
        if (quantizedWeights_.data.empty())
        {
            throw std::runtime_error(
                "INT8 weights have not been quantized"
            );
        }

        const auto& inputShape = input.shape();
        const auto& weightShape = quantizedWeights_.shape;

        size_t M = inputShape[0];
        size_t K = inputShape[1];
        size_t N = weightShape[1];

        Tensor output({M, N});

        for (size_t row = 0; row < M; ++row)
        {
            for (size_t col = 0; col < N; ++col)
            {
                float sum = 0.0f;

                for (size_t k = 0; k < K; ++k)
                {
                    int8_t qWeight =
                        quantizedWeights_.data[k * N + col];

                    float dequantizedWeight =
                        static_cast<float>(qWeight) *
                        quantizedWeights_.scale;

                    sum +=
                        input.at(row, k) *
                        dequantizedWeight;
                }

                output.at(row, col) = sum;
            }
        }

        size_t rows = output.shape()[0];
        size_t cols = output.shape()[1];

        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {
                output.at(row, col) += bias_.at(0, col);
            }
        }

        return output;
    }

    Tensor output = useCuda
        ? cudaMatmul(input, weights_)
        : input.matmul(weights_);

    size_t rows = output.shape()[0];
    size_t cols = output.shape()[1];

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            output.at(row, col) += bias_.at(0, col);
        }
    }

    return output;
}

 //output = input x weight + bias calc