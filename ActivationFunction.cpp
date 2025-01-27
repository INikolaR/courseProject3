#include "ActivationFunction.h"

#include <cassert>
#include <cmath>
#include <functional>

namespace neural_network {

ActivationFunction ActivationFunction::ReLU() {
    return ActivationFunction([](double x) { return (x > 0) * x; },
                              [](double x) { return (x > 0); }, "ReLU()");
}

ActivationFunction ActivationFunction::LeakyReLU() {
    return ActivationFunction(
        [](double x) {
            return (x > 0) * (1 - LeakyReLUCoefficient) * x +
                   LeakyReLUCoefficient * x;
        },
        [](double x) {
            return (x > 0) * (1 - LeakyReLUCoefficient) + LeakyReLUCoefficient;
        },
        "LeakyReLU()");
}

ActivationFunction ActivationFunction::Sigmoid() {
    return ActivationFunction([](double x) { return 1 / (1 + exp(-x)); },
                              [](double x) {
                                  double s = 1 / (1 + exp(-x));
                                  return s * (1 - s);
                              },
                              "Sigmoid()");
}

ActivationFunction ActivationFunction::Id() {
    return ActivationFunction([](double x) { return x; },
                              [](double x) { return 1; }, "Id()");
}

ActivationFunction::ActivationFunction(std::function<double(double)>&& f0,
                                       std::function<double(double)>&& f1)
    : f0_(std::move(f0)),
      f1_(std::move(f1)),
      description_(std::move("CustomFunction")) {
}

ActivationFunction::ActivationFunction(std::function<double(double)>&& f0,
                                       std::function<double(double)>&& f1,
                                       std::string description)
    : f0_(std::move(f0)),
      f1_(std::move(f1)),
      description_(std::move(description)) {
}

double ActivationFunction::evaluate0(double value) const {
    assert(f0_);
    return f0_(value);
}

double ActivationFunction::evaluate1(double value) const {
    assert(f1_);
    return f1_(value);
}

Vector ActivationFunction::evaluate0(const Vector& x) const {
    assert(f0_);
    Vector x_result;
    x_result.reserve(x.size() + 1);
    for (double element : x) {
        x_result.emplace_back(f0_(element));
    }
    return x_result;
}

Vector ActivationFunction::evaluate1(const Vector& x) const {
    assert(f1_);
    Vector x_result;
    x_result.reserve(x.size() + 1);
    for (double element : x) {
        x_result.emplace_back(f1_(element));
    }
    return x_result;
}

std::string ActivationFunction::describe() const {
    return description_;
}

}  // namespace neural_network
