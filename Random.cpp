#include "Random.h"

#include <algorithm>
#include <cassert>

namespace neural_network {

Random::Random() : engine_(std::mt19937(Seed)) {
}

Matrix Random::givensAngleMatrix(size_t rows, size_t cols) {
    assert(rows <= cols);
    Matrix generated;
    generated.reserve(rows);
    std::uniform_real_distribution<double> uniform_d{-std::acos(-1),
                                                     std::acos(-1)};
    for (size_t i = 0; i < rows; ++i) {
        Vector v(cols - i);
        std::generate(v.begin(), v.end(), [&](){ return uniform_d(engine_); });
        generated.emplace_back(v);
    }
    return generated;
}

Vector Random::singularValues(size_t length) {
    std::normal_distribution<double> normal_d;
    Vector generated(length, normal_d(engine_));
    std::sort(generated.begin(), generated.end());
    return generated;
}

}  // namespace neural_network
