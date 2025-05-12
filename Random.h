#pragma once
#include <random>

#include "CustomTypes.h"

namespace neural_network {
class Random {
public:
    Random();
    Random(int seed);
    std::vector<double> normal(In in, Out out);
    std::vector<double> kaiming(In in, Out out);
    std::vector<double> xavier(In in, Out out);

private:
    static constexpr int Seed = 1234567;
    std::mt19937 engine_;
};
}  // namespace neural_network
