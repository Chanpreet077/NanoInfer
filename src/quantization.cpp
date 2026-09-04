#include "quantization.h"

#include <algorithm>
#include <cmath>

QuantizedTensor quantizeTensor(const Tensor& tensor)
{
    QuantizedTensor result;

    result.shape = tensor.shape();

    size_t totalElements = 1;

    for (size_t dim : result.shape)
    {
        totalElements *= dim;
    }

    float maxAbs = 0.0f;

    for (size_t i = 0; i < totalElements; ++i)
    {
        float value = std::abs(tensor[i]);

        if (value > maxAbs)
        {
            maxAbs = value;
        }
    }

    if (maxAbs == 0.0f)
    {
        result.scale = 1.0f;
    }
    else
    {
        result.scale = maxAbs / 127.0f;
    }

    result.data.resize(totalElements);

    for (size_t i = 0; i < totalElements; ++i)
    {
        float scaled = tensor[i] / result.scale;

        int quantizedValue =
            static_cast<int>(std::round(scaled));

        quantizedValue =
            std::max(-127, std::min(127, quantizedValue));

        result.data[i] =
            static_cast<int8_t>(quantizedValue);
    }

    return result;
}