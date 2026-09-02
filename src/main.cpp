#include "model.h"
#include "model_loader.h"

#include <iostream>
#include <vector>

int main()
{
    Model model;

    model.load("models/mnist_weights.txt");

    // Load the 784 pixel values from the exported MNIST test image
    std::vector<float> imageValues =
        loadWeights("models/mnist_test_image.txt", 784);

    Tensor input({1, 784});
    input.loadData(imageValues);

    // Run inference through NanoInfer
    Tensor probabilities = model.forward(input);

    int predictedClass = 0;
    float highestProbability = probabilities[0];

    for (int i = 1; i < 10; i++) {
        if (probabilities[i] > highestProbability) {
            highestProbability = probabilities[i];
            predictedClass = i;
        }
    }

    std::cout << "NanoInfer prediction: "
              << predictedClass
              << std::endl;

    std::cout << "Probabilities: ";
    probabilities.print();

    return 0;
}