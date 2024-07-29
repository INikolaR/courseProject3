#include "ActivationFunction.h"

#include <cassert>
#include <cmath>
#include <functional>

namespace neural_network {

ActivationFunction ActivationFunction::ReLU() {
    return ActivationFunction([](double x) { return (x > 0) * x; },
                              [](double x) { return (x > 0); });
}

ActivationFunction ActivationFunction::LeakyReLU() {
    return ActivationFunction(
        [](double x) {
            return (x > 0) * (1 - LeakyReLUCoefficient) * x +
                   LeakyReLUCoefficient * x;
        },
        [](double x) {
            return (x > 0) * (1 - LeakyReLUCoefficient) + LeakyReLUCoefficient;
        });
}

ActivationFunction ActivationFunction::Sigmoid() {
    return ActivationFunction([](double x) { return 1 / (1 + exp(-x)); },
                              [](double x) {
                                  double s = 1 / (1 + exp(-x));
                                  return s * (1 - s);
                              });
}

ActivationFunction::ActivationFunction(std::function<double(double)>&& f0,
                                       std::function<double(double)>&& f1)
    : f0_(std::move(f0)), f1_(std::move(f1)) {
}

double ActivationFunction::evaluate0(double value) const {
    assert(f0_);
    return f0_(value);
}

double ActivationFunction::evaluate1(double value) const {
    assert(f1_);
    return f1_(value);
}

std::vector<double> ActivationFunction::evaluate0(
    const std::vector<double>& x) const {
    assert(f0_);
    std::vector<double> x_result;
    x_result.reserve(x.size() + 1);
    for (double element : x) {
        x_result.emplace_back(f0_(element));
    }
    return x_result;
}

std::vector<double> ActivationFunction::evaluate1(
    const std::vector<double>& x) const {
    assert(f1_);
    std::vector<double> x_result;
    x_result.reserve(x.size() + 1);
    for (double element : x) {
        x_result.emplace_back(f1_(element));
    }
    return x_result;
}

}  // namespace neural_network
