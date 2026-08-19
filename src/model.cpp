#include "model.h"
#include "model_loader.h"

#include <vector>

Model::Model()
    : layer1_(2, 4),
      layer2_(4, 2)
{
}

void Model::load(const std::string& filename)
{
    std::vector<float> values = loadWeights(filename, 22);

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

Tensor Model::forward(const Tensor& input) const
{
    Tensor hidden = layer1_.forward(input);
    Tensor activated = relu_.forward(hidden);
    Tensor output = layer2_.forward(activated);

    return softmax_.forward(output);
}