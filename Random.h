#pragma once
#include <random>

#include "CustomTypes.h"

namespace neural_network {
class Random {
public:
    Random();
    Matrix givensAngleMatrix(size_t rows, size_t cols);
    Vector singularValues(size_t length);
private:
    static constexpr int Seed = 123;
    std::mt19937 engine_;
};
}  // namespace neural_network
