#include "tensor.h"
#include "linear.h"
#include "relu.h"
#include "softmax.h"

#include <iostream>

int main()
{
    Tensor input({1, 2});

    input.at(0, 0) = 2;
    input.at(0, 1) = -3;

    // First layer: 2 inputs -> 3 hidden values

    Linear layer1(2, 3);

    layer1.weights().loadData({
        1, -2, 3,
        4,  1, -1
    });

    layer1.bias().loadData({
        0, 0, 0
    });

    ReLU relu;

    // Second layer: 3 hidden values -> 2 outputs
    Linear layer2(3, 2);

    layer2.weights().loadData({
        1,  2,
        -1,  1,
        2, -1
    });

    layer2.bias().loadData({
        1, 1
    });


    Tensor hidden = layer1.forward(input);
    Tensor activated = relu.forward(hidden);
    Tensor output = layer2.forward(activated);
    Softmax softmax;

    Tensor probabilities = softmax.forward(output);
    
    std::cout << "Hidden: ";
    hidden.print();

    std::cout << "After ReLU: ";
    activated.print();


    std::cout << "Final output: ";
    output.print();

    std::cout << "Probabilities: ";
    probabilities.print();

    return 0;
}