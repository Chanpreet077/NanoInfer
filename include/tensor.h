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
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    Tensor add(const Tensor& other) const;
    Tensor multiply(float scalar) const;
    Tensor matmul(const Tensor& other) const;
    void loadData(const std::vector<float>& values);
};

#endif