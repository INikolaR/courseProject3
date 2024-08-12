#include "LossFunction.h"
#include "VectorOperations.h"

namespace neural_network {

LossFunction LossFunction::Euclid() {
    return LossFunction(
        [](const Vector& x, const Vector& y) {
            Vector d = x - y;
            return dot(d, d);
        },
        [](const Vector& x, const Vector& y) {
            return 2 * (x - y);
        });
}

LossFunction::LossFunction(
    std::function<double(const Vector&, const Vector&)>&& f0,
    std::function<Vector(const Vector&, const Vector&)>&& f1) : f0_(f0), f1_(f1) {
}

double LossFunction::evaluate0(const Vector& x, const Vector& y) const {
    return f0_(x, y);
}

Vector LossFunction::evaluate1(const Vector& x, const Vector& y) const {
    return f1_(x, y);
}

}  // namespace neural_network
