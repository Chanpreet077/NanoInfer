#include "model.h"
#include "model_loader.h"

#include <vector>


Model::Model()
    : layer1_(784, 128),
      layer2_(128, 10)
{
}
/*
void Model::load(const std::string& filename) OLDDDDDD //manually loading weights + biases
{
    std::vector<float> values = loadWeights(filename, 101770);

    layer1_.weights().loadData({
        values[0], values[1],
        values[2], values[3],
        values[4], values[5],
        values[6], values[7]
    });

    layer1_.bias().loadData({
        values[8],
        values[9],
        values[10],
        values[11]
    });

    layer2_.weights().loadData({
        values[12], values[13],
        values[14], values[15],
        values[16], values[17],
        values[18], values[19]
    });

    layer2_.bias().loadData({
        values[20],
        values[21]
    });
}
    */

void Model::load(const std::string& filename)
{
    std::vector<float> values = loadWeights(filename, 101770);

    size_t index = 0;

    // Layer 1 weights: 784 x 128 = 100352 values
    std::vector<float> layer1Weights(
        values.begin() + index,
        values.begin() + index + 100352
    );

    index += 100352;

    // Layer 1 bias: 128 values
    std::vector<float> layer1Bias(
        values.begin() + index,
        values.begin() + index + 128
    );

    index += 128;

    // Layer 2 weights: 128 x 10 = 1280 values
    std::vector<float> layer2Weights(
        values.begin() + index,
        values.begin() + index + 1280
    );

    index += 1280;

    // Layer 2 bias: 10 values
    std::vector<float> layer2Bias(
        values.begin() + index,
        values.begin() + index + 10
    );

    layer1_.weights().loadData(layer1Weights);
    layer1_.bias().loadData(layer1Bias);

    layer2_.weights().loadData(layer2Weights);
    layer2_.bias().loadData(layer2Bias);

    // Create INT8 copies of the weights after FP32 weights are loaded
    layer1_.quantizeWeights();
    layer2_.quantizeWeights();
}

Tensor Model::forward(
    const Tensor& input,
    bool useCuda,
    bool useInt8
) const
{
    Tensor hidden =
        layer1_.forward(input, useCuda, useInt8);

    Tensor activated =
        relu_.forward(hidden);

    Tensor output =
        layer2_.forward(activated, useCuda, useInt8);

    return softmax_.forward(output);
}