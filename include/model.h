#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include "linear.h"
#include "relu.h"
#include "softmax.h"

#include <string>

class Model {
private:
    Linear layer1_;
    Linear layer2_;
    ReLU relu_;
    Softmax softmax_;

public:
    Model();

    void load(const std::string& filename);

    Tensor forward(const Tensor& input) const;
};

#endif