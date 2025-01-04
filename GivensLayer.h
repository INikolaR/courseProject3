#pragma once
#include "ActivationFunction.h"
#include "CustomTypes.h"
#include "Random.h"

namespace neural_network {

class GivensLayer {
public:
    GivensLayer(const Vector& weights, size_t in, size_t out);
    GivensLayer(Random& rnd, size_t in, size_t out);
    size_t sizeIn() const;
    size_t sizeOut() const;
    Vector forward(const Vector& x) const;
    Vector forwardOnTrain(const Vector& x) const;
    Vector backwardCalcGradient(Vector& u, const Vector& x, Vector& z) const;
    void update(const Vector& grad, double step);

private:
    GivensLayer(const SVD& svd, size_t in, size_t out);

    size_t n_;
    size_t m_;
    size_t min_n_m_;
    Vector alpha_;
    Vector beta_;
    Vector sigma_;
};

}  // namespace neural_network
