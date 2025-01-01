#pragma once
#include "CustomTypes.h"

namespace neural_network {
class MatrixLayer {
public:
    MatrixLayer(const Vector& weights, size_t in, size_t out);
    size_t sizeIn() const;
    size_t sizeOut() const;
    Vector forward(const Vector& x) const;
    Vector forwardOnTrain(const Vector& x) const;
    Vector backwardCalcGradient(Vector& u, const Vector& x, Vector& z) const;
    void update(const Vector& grad, double step);

private:
    size_t n_;
    size_t m_;
    Vector w_;
};
}  // namespace neural_network
