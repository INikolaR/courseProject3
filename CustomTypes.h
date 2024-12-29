#pragma once
#include <vector>

namespace neural_network {
using Vector = std::vector<double>;
struct TrainUnit {
    Vector x;
    Vector y;
};
struct SVD {
    Vector U;
    Vector sigma;
    Vector V;
};
}  // namespace neural_network
