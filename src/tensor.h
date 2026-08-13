#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>

class Tensor {
private:
    std::vector<float> data_;
    std::vector<size_t> shape_;

public:
    Tensor(const std::vector<size_t>& shape);

    size_t size() const;

    const std::vector<size_t>& shape() const;

    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    void print() const;
};

#endif