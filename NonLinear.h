#pragma once
#include <functional>
#include <vector>

#include "VectorOperations.h"

namespace neural_network {

class NonLinear {
public:
    static NonLinear ReLU();
    static NonLinear LeakyReLU(double coeff = LeakyReLUCoefficient);
    static NonLinear Sigmoid();
    static NonLinear Id();

    NonLinear(std::function<double(double)>&& f0,
              std::function<double(double)>&& f1);
    NonLinear(std::function<double(double)>&& f0,
              std::function<double(double)>&& f1, std::string description);

    double evaluate0(double x) const;
    double evaluate1(double x) const;
    Matrix evaluate0(const Matrix& x) const;
    Matrix evaluate1(const Matrix& x) const;
    std::string describe() const;

private:
    static constexpr double LeakyReLUCoefficient = 0.1;

    std::function<double(double)> f0_;
    std::function<double(double)> f1_;
    std::string description_;
};

}  // namespace neural_network
