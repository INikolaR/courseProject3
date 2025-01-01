#pragma once
#include <random>

#include "CustomTypes.h"

namespace neural_network {
class Random {
public:
    Random();
    void givensAngles(Vector::iterator begin, Vector::iterator end);
    void singularValues(Vector::iterator begin, Vector::iterator end);
    Vector kaiming(size_t in, size_t out);

private:
    static constexpr int Seed = 1234567;
    std::mt19937 engine_;
};
}  // namespace neural_network
