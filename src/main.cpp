#include "model.h"
#include <iostream>

int main()
{
    Model model;

    model.load("models/xor_weights.txt");

    float testInputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    for (int i = 0; i < 4; i++) {
        Tensor input({1, 2});

        input.at(0, 0) = testInputs[i][0];
        input.at(0, 1) = testInputs[i][1];

        Tensor probabilities = model.forward(input);

        int predictedClass =
            probabilities[0] > probabilities[1] ? 0 : 1;

        std::cout
            << "Input: ["
            << testInputs[i][0]
            << ", "
            << testInputs[i][1]
            << "] -> Prediction: "
            << predictedClass
            << std::endl;
    }

    return 0;
}