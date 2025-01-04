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
    size_t n_;
    size_t m_;
    size_t min_n_m_;
    Vector w_;
    Vector::iterator alpha_;
    Vector::iterator sigma_;
    Vector::iterator beta_;
};

}  // namespace neural_network
