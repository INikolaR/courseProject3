#include "Random.h"

#include <algorithm>
#include <cassert>

#include "time.h"

namespace neural_network {

Random::Random() : engine_(std::mt19937(Seed)) {
}

Random::Random(int seed) : engine_(std::mt19937(seed)) {
}

std::vector<double> Random::normal(In in, Out out) {
    std::normal_distribution<double> normal_d{0, 1};
    std::vector<double> generated((in + 1) * out);
    std::generate(generated.begin(), generated.end(),
                  [&]() { return normal_d(engine_); });
    return generated;
}

std::vector<double> Random::kaiming(In in, Out out) {
    std::normal_distribution<double> normal_d{
        0, sqrt(2 / static_cast<double>(in))};
    std::vector<double> generated((in + 1) * out);
    std::generate(generated.begin(), generated.end(),
                  [&]() { return normal_d(engine_); });
    return generated;
}

std::vector<double> Random::xavier(In in, Out out) {
    std::uniform_real_distribution<double> u_d{
        -sqrt(6 / static_cast<double>(in + out)),
        sqrt(6 / static_cast<double>(in + out))};
    std::vector<double> generated((in + 1) * out);
    std::generate(generated.begin(), generated.end(),
                  [&]() { return u_d(engine_); });
    return generated;
}

}  // namespace neural_network
