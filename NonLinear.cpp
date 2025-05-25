#include "NonLinear.h"

#include <cassert>
#include <cmath>

namespace neural_network {

NonLinear NonLinear::ReLU() {
    return NonLinear([](double x) { return (x > 0) * x; },
                     [](double x) { return (x > 0); }, "ReLU()");
}

NonLinear NonLinear::LeakyReLU(double coeff) {
    return NonLinear(
        [](double x) {
            return (x > 0) * (1 - LeakyReLUCoefficient) * x +
                   LeakyReLUCoefficient * x;
        },
        [](double x) {
            return (x > 0) * (1 - LeakyReLUCoefficient) + LeakyReLUCoefficient;
        },
        "LeakyReLU()");
}

NonLinear NonLinear::Sigmoid() {
    return NonLinear([](double x) { return 1 / (1 + exp(-x)); },
                     [](double x) {
                         double s = 1 / (1 + exp(-x));
                         return s * (1 - s);
                     },
                     "Sigmoid()");
}

NonLinear NonLinear::Id() {
    return NonLinear([](double x) { return x; }, [](double x) { return 1; },
                     "Id()");
}

NonLinear::NonLinear(std::function<double(double)>&& f0,
                     std::function<double(double)>&& f1)
    : f0_(std::move(f0)),
      f1_(std::move(f1)),
      description_(std::move("CustomFunction")) {
    assert(f0_ && "f0 should be not null");
    assert(f1_ && "f1 should be not null");
}

NonLinear::NonLinear(std::function<double(double)>&& f0,
                     std::function<double(double)>&& f1,
                     std::string description)
    : f0_(std::move(f0)),
      f1_(std::move(f1)),
      description_(std::move(description)) {
    assert(f0_ && "f0 should be not null");
    assert(f1_ && "f1 should be not null");
}

double NonLinear::evaluate0(double value) const {
    assert(f0_);
    return f0_(value);
}

double NonLinear::evaluate1(double value) const {
    assert(f1_);
    return f1_(value);
}

Matrix NonLinear::evaluate0(const Matrix& x) const {
    assert(f0_);
    return x.unaryExpr(f0_);
}

Matrix NonLinear::evaluate1(const Matrix& x) const {
    assert(f1_);
    return x.unaryExpr(f1_);
}

std::string NonLinear::describe() const {
    return description_;
}

}  // namespace neural_network
