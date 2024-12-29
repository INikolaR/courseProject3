#pragma once
#include <random>

#include "CustomTypes.h"

namespace neural_network {
class Random {
public:
    Random();
    Vector givensAngles(size_t size);
    Vector singularValues(size_t length);

private:
    static constexpr int Seed = 1234567;
    std::mt19937 engine_;
};
}  // namespace neural_network
