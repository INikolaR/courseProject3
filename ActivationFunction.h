#pragma once
#include <functional>
#include <vector>

#include "VectorOperations.h"

namespace neural_network {

class ActivationFunction {
public:
    static ActivationFunction ReLU();
    static ActivationFunction LeakyReLU();
    static ActivationFunction Sigmoid();
    static ActivationFunction Id();
    ActivationFunction(std::function<double(double)>&& f0,
                       std::function<double(double)>&& f1);
    double evaluate0(double x) const;
    double evaluate1(double x) const;
    Vector evaluate0(const Vector& x) const;
    Vector evaluate1(const Vector& x) const;

private:
    static constexpr double LeakyReLUCoefficient = 0.1;

    const std::function<double(double)> f0_;
    const std::function<double(double)> f1_;
};

}  // namespace neural_network
