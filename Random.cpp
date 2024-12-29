#include "Random.h"

#include <algorithm>
#include <cassert>

#include "time.h"

namespace neural_network {

Random::Random() : engine_(std::mt19937(Seed)) {
}

Vector Random::givensAngles(size_t size) {
    Vector generated(size);
    std::normal_distribution<double> norm_d{std::acos(-1) / 2, 0.2};
    std::generate(generated.begin(), generated.end(),
                  [&]() { return norm_d(engine_); });
    return generated;
}

Vector Random::singularValues(size_t length) {
    std::normal_distribution<double> normal_d;
    Vector generated(length);
    std::generate(generated.begin(), generated.end(),
                  [&]() { return normal_d(engine_) + 1; });
    std::sort(generated.begin(), generated.end());
    return generated;
}

}  // namespace neural_network
