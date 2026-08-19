#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>
#include <vector>

std::vector<float> loadWeights(
    const std::string& filename,
    size_t count
);

#endif