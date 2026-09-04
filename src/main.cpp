#include "model.h"
#include "model_loader.h"

#include <iostream>
#include <vector>
#include <chrono>

int main()
{
    Model model;

    /* Small INT8 quantization correctness test
    Tensor testTensor({1, 4});

    testTensor.at(0, 0) = -1.0f;
    testTensor.at(0, 1) = -0.5f;
    testTensor.at(0, 2) = 0.5f;
    testTensor.at(0, 3) = 1.0f;

    QuantizedTensor quantized = quantizeTensor(testTensor);

    std::cout << "INT8 quantization test:\n";
    std::cout << "Scale: " << quantized.scale << "\n";

    for (size_t i = 0; i < quantized.data.size(); ++i)
    {
        int q = static_cast<int>(quantized.data[i]);
        float reconstructed = q * quantized.scale;

        std::cout << "INT8: "
                  << q
                  << " -> reconstructed: "
                  << reconstructed
                  << "\n";
    }

    std::cout << "\n"; */


    /* Small CPU vs CUDA matmul correctness test
    Tensor a({2, 3});
    Tensor b({3, 2});

    a.at(0, 0) = 1;
    a.at(0, 1) = 2;
    a.at(0, 2) = 3;
    a.at(1, 0) = 4;
    a.at(1, 1) = 5;
    a.at(1, 2) = 6;

    b.at(0, 0) = 7;
    b.at(0, 1) = 8;
    b.at(1, 0) = 9;
    b.at(1, 1) = 10;
    b.at(2, 0) = 11;
    b.at(2, 1) = 12;

    Tensor cpuResult = a.matmul(b);
    Tensor gpuResult = cudaMatmul(a, b);

    std::cout << "CPU result:\n";

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            std::cout << cpuResult.at(i, j) << " ";
        }

        std::cout << "\n";
    }

    std::cout << "\nGPU result:\n";

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            std::cout << gpuResult.at(i, j) << " ";
        }

        std::cout << "\n";
    }

    std::cout << "\n"; */

    model.load("models/mnist_weights.txt");

    // Load the 784 pixel values from the exported MNIST test image
    std::vector<float> imageValues =
        loadWeights("models/mnist_test_image.txt", 784);

    Tensor input({1, 784});
    input.loadData(imageValues);

    // Run inference through NanoInfer using CPU FP32
    Tensor cpuProbabilities = model.forward(input, false, false);

    // Run inference through NanoInfer using CUDA FP32
    Tensor gpuProbabilities = model.forward(input, true, false);

    // Run inference through NanoInfer using INT8 weights
    Tensor int8Probabilities = model.forward(input, false, true);

    int cpuPredictedClass = 0;
    float cpuHighestProbability = cpuProbabilities[0];

    int gpuPredictedClass = 0;
    float gpuHighestProbability = gpuProbabilities[0];

    int int8PredictedClass = 0;
    float int8HighestProbability = int8Probabilities[0];

for (int i = 1; i < 10; i++) {
    if (cpuProbabilities[i] > cpuHighestProbability) {
        cpuHighestProbability = cpuProbabilities[i];
        cpuPredictedClass = i;
    }

    if (gpuProbabilities[i] > gpuHighestProbability) {
        gpuHighestProbability = gpuProbabilities[i];
        gpuPredictedClass = i;
    }

    if (int8Probabilities[i] > int8HighestProbability) {
        int8HighestProbability = int8Probabilities[i];
        int8PredictedClass = i;
    }
}

    std::cout << "CPU NanoInfer prediction: "
              << cpuPredictedClass
              << std::endl;

    std::cout << "CPU probabilities: ";
    cpuProbabilities.print();

    std::cout << "GPU NanoInfer prediction: "
              << gpuPredictedClass
              << std::endl;

    std::cout << "GPU probabilities: ";
    gpuProbabilities.print();

    std::cout << "INT8 NanoInfer prediction: "
          << int8PredictedClass
          << std::endl;

    std::cout << "INT8 probabilities: ";
    int8Probabilities.print();

        // Compare FP32 vs INT8 weight memory usage

    size_t totalWeights =
        (784 * 128) +
        (128 * 10);

    size_t fp32WeightBytes =
        totalWeights * sizeof(float);

    size_t int8WeightBytes =
        totalWeights * sizeof(int8_t);

    double memoryReductionPercent =
        (1.0 - static_cast<double>(int8WeightBytes) /
            static_cast<double>(fp32WeightBytes))
        * 100.0;

    std::cout << "\n--- Weight Memory ---\n";

    std::cout << "Total weights: "
            << totalWeights
            << "\n";

    std::cout << "FP32 weight memory: "
            << fp32WeightBytes
            << " bytes\n";

    std::cout << "INT8 weight memory: "
            << int8WeightBytes
            << " bytes\n";

    std::cout << "Memory reduction: "
            << memoryReductionPercent
            << "%\n";

    // Benchmark CPU vs CUDA inference

    const int runs = 1000;

    // Warm up CUDA
    for (int i = 0; i < 10; ++i)
    {
        model.forward(input, true);
    }

    // CPU benchmark
    auto cpuStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; ++i)
    {
        model.forward(input, false);
    }

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    double cpuTotalMs =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart
        ).count();

    double cpuAvgMs = cpuTotalMs / runs;


    // CUDA benchmark
    auto gpuStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; ++i)
    {
        model.forward(input, true);
    }

    auto gpuEnd = std::chrono::high_resolution_clock::now();

    double gpuTotalMs =
        std::chrono::duration<double, std::milli>(
            gpuEnd - gpuStart
        ).count();

    double gpuAvgMs = gpuTotalMs / runs;


    std::cout << "\n--- Benchmark ---\n";

    std::cout << "Runs: "
              << runs
              << "\n";

    std::cout << "CPU average inference time: "
              << cpuAvgMs
              << " ms\n";

    std::cout << "CUDA average inference time: "
              << gpuAvgMs
              << " ms\n";

    std::cout << "CPU/CUDA ratio: "
              << cpuAvgMs / gpuAvgMs
              << "x\n";

    return 0;
}