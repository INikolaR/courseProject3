#include "Random.h"

#include <algorithm>
#include <cassert>

#include "time.h"

namespace neural_network {

Random::Random() : engine_(std::mt19937(Seed)) {
}

void Random::givensAngles(Vector::iterator begin, Vector::iterator end) {
    std::normal_distribution<double> norm_d{std::acos(-1) / 2, 0.2};
    std::generate(begin, end, [&]() { return norm_d(engine_); });
}

void Random::singularValues(Vector::iterator begin, Vector::iterator end) {
    std::normal_distribution<double> normal_d;
    std::generate(begin, end, [&]() { return normal_d(engine_) + 1; });
    std::sort(begin, end);
}

Vector Random::normal(size_t in, size_t out) {
    std::normal_distribution<double> normal_d{0, 1};
    Vector generated((in + 1) * out);
    std::generate(generated.begin(), generated.end(),
                  [&]() { return normal_d(engine_); });
    return generated;
}

Vector Random::kaiming(size_t in, size_t out) {
    std::normal_distribution<double> normal_d{
        0, sqrt(2 / static_cast<double>(in))};
    Vector generated((in + 1) * out);
    std::generate(generated.begin(), generated.end(),
                  [&]() { return normal_d(engine_); });
    return generated;
}

Vector Random::xavier(size_t in, size_t out) {
    std::uniform_real_distribution<double> u_d{
        -sqrt(6 / static_cast<double>(in + out)),
        sqrt(6 / static_cast<double>(in + out))};
    Vector generated((in + 1) * out);
    std::generate(generated.begin(), generated.end(),
                  [&]() { return u_d(engine_); });
    return generated;
}

}  // namespace neural_network
