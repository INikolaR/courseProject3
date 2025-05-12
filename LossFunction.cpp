#include "LossFunction.h"

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

LossFunction::LossFunction(
    std::function<double(const Matrix&, const Matrix&)>&& f0,
    std::function<Matrix(const Matrix&, const Matrix&)>&& f1)
    : f0_(f0), f1_(f1) {
}

double LossFunction::evaluate0(const Matrix& x, const Matrix& y) const {
    return f0_(x, y);
}

Matrix LossFunction::evaluate1(const Matrix& x, const Matrix& y) const {
    return f1_(x, y);
}

}  // namespace neural_network
