#pragma once
#include <functional>

#include "CustomTypes.h"

namespace neural_network {

class LossFunction {
    using DistanceFunc = std::function<double(const Matrix&, const Matrix&)>;
    using GradientFunc = std::function<Matrix(const Matrix&, const Matrix&)>;

public:
    static LossFunction Euclid();
    static LossFunction Manhattan();

    LossFunction(DistanceFunc f0, GradientFunc f1);

    double evaluate0(const Matrix& x, const Matrix& y) const;
    Matrix evaluate1(const Matrix& x, const Matrix& y) const;

private:
    DistanceFunc f0_;
    GradientFunc f1_;
};

}  // namespace neural_network
