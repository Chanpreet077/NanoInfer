#include "tensor.h"
#include <stdexcept>

#include <iostream>

Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape)
{
    size_t totalSize = 1;

    for (size_t dimension : shape) {
        totalSize *= dimension;
    }

    data_.resize(totalSize, 0.0f);
}

size_t Tensor::size() const
{
    return data_.size();
}

const std::vector<size_t>& Tensor::shape() const
{
    return shape_;
}

float& Tensor::operator[](size_t index)
{
    return data_[index];
}

const float& Tensor::operator[](size_t index) const
{
    return data_[index];
}

Tensor Tensor::add(const Tensor& other) const
{
    Tensor result(shape_);

    for (size_t i = 0; i < data_.size(); i++) {
        result[i] = data_[i] + other[i];
    }

    return result;
}

Tensor Tensor::multiply(float scalar) const
{
    Tensor result(shape_);

    for (size_t i = 0; i < data_.size(); i++) {
        result[i] = data_[i] * scalar;
    }

    return result;
}

Tensor Tensor::matmul(const Tensor& other) const
{
    size_t aRows = shape_[0];
    size_t aCols = shape_[1];

    size_t bRows = other.shape_[0];
    size_t bCols = other.shape_[1];

    Tensor result({aRows, bCols});

    for (size_t row = 0; row < aRows; row++) {
        for (size_t col = 0; col < bCols; col++) {

            float sum = 0.0f;

            for (size_t k = 0; k < aCols; k++) {
                sum += at(row, k) * other.at(k, col);
            }

            result.at(row, col) = sum;
        }
    }

    return result;
}

void Tensor::loadData(const std::vector<float>& values)
{
    if (values.size() != data_.size()) {
        throw std::runtime_error("Data size does not match tensor size");
    }

    data_ = values;
}

void Tensor::print() const
{
    for (float value : data_) {
        std::cout << value << " ";
    }

    std::cout << std::endl;
}

float& Tensor::at(size_t row, size_t col)
{
    size_t columns = shape_[1];

    size_t index = row * columns + col;

    return data_[index];
}

const float& Tensor::at(size_t row, size_t col) const
{
    size_t columns = shape_[1];

    size_t index = row * columns + col;

    return data_[index];
}