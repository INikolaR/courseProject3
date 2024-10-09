#pragma once
#include <vector>

namespace neural_network {
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double> >;
struct TrainUnit {
    Vector x;
    Vector y;
};
struct Gradient {
    Matrix U;
    Vector sigma;
    Matrix V;
};
}  // namespace neural_network
