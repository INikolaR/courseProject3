#pragma once
#include <functional>
#include <vector>

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
    std::vector<double> evaluate0(const std::vector<double>& x) const;
    std::vector<double> evaluate1(const std::vector<double>& x) const;

private:
    static constexpr double LeakyReLUCoefficient = 0.1;

    const std::function<double(double)> f0_;
    const std::function<double(double)> f1_;
};

}  // namespace neural_network
