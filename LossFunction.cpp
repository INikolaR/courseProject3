#include "LossFunction.h"

#include <cassert>

#include "VectorOperations.h"

namespace neural_network {
LossFunction LossFunction::Euclid() {
    return LossFunction(
        [](const Matrix& x, const Matrix& y) {
            Matrix d = x - y;
            return (x - y).squaredNorm();
        },
        [](const Matrix& x, const Matrix& y) { return 2 * (x - y); });
}

LossFunction LossFunction::Manhattan() {
    return LossFunction(
        [](const Matrix& x, const Matrix& y) {
            return (x - y).cwiseAbs().sum();
        },
        [](const Matrix& x, const Matrix& y) { return (x - y).cwiseSign(); });
}

LossFunction::LossFunction(DistanceFunc f0, GradientFunc f1)
    : f0_(std::move(f0)), f1_(std::move(f1)) {
    assert(f0_ && "f0 should be not null");
    assert(f1_ && "f1 should be not null");
}

double LossFunction::evaluate0(const Matrix& x, const Matrix& y) const {
    assert(x.rows() == y.rows() && x.cols() == y.cols() &&
           "different sizes of x and y");
    return f0_(x, y);
}

Matrix LossFunction::evaluate1(const Matrix& x, const Matrix& y) const {
    assert(x.rows() == y.rows() && x.cols() == y.cols() &&
           "different sizes of x and y");
    return f1_(x, y);
}

}  // namespace neural_network
