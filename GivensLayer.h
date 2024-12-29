#pragma once
#include "ActivationFunction.h"
#include "CustomTypes.h"
#include "Random.h"

namespace neural_network {

class GivensLayer {
public:
    GivensLayer(size_t in, size_t out);
    size_t sizeIn() const;
    size_t sizeOut() const;
    Vector forward(const Vector& x) const;
    Vector forwardWithoutShrinking(const Vector& x) const;
    SVD backwardCalcGradient(Vector& u, Vector& z) const;
    void update(const SVD& grad, double step);

private:
    Random rnd_;
    size_t n_;
    size_t m_;
    size_t min_n_m_;
    Vector alpha_;
    Vector beta_;
    Vector sigma_;
};

}  // namespace neural_network
