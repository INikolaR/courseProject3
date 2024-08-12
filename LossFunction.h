#pragma once
#include <functional>

#include "CustomTypes.h"

namespace neural_network {

class LossFunction {
public:
    static LossFunction Euclid();
    LossFunction(std::function<double(const Vector&, const Vector&)>&& f0,
                 std::function<Vector(const Vector&, const Vector&)>&& f1);
    double evaluate0(const Vector& x, const Vector& y) const;
    Vector evaluate1(const Vector& x, const Vector& y) const;

private:
    std::function<double(const Vector&, const Vector&)> f0_;
    std::function<Vector(const Vector&, const Vector&)> f1_;
};

}  // namespace neural_network
