#include "model_loader.h"

#include <fstream>
#include <stdexcept>

std::vector<float> loadWeights(  //returns all weights + biases from file as one large vector called values
    const std::string& filename,
    size_t count
)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open weights file");
    }

    std::vector<float> values;

    float value;

    while (file >> value) {
        values.push_back(value);
    }

   if (values.size() != count) {
    throw std::runtime_error(
        "Incorrect number of weights in file. Expected " +
        std::to_string(count) +
        ", got " +
        std::to_string(values.size())
    );
}
    return values;
}