#pragma once
#include <functional>

#include "CustomTypes.h"

namespace neural_network {

class LossFunction {
public:
    static LossFunction Euclid();
    static LossFunction Manhattan();

    LossFunction(std::function<double(const Matrix&, const Matrix&)>&& f0,
                 std::function<Matrix(const Matrix&, const Matrix&)>&& f1);

    double evaluate0(const Matrix& x, const Matrix& y) const;
    Matrix evaluate1(const Matrix& x, const Matrix& y) const;

private:
    std::function<double(const Matrix&, const Matrix&)> f0_;
    std::function<Matrix(const Matrix&, const Matrix&)> f1_;
};

}  // namespace neural_network
