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
    Matrix passForward(const Matrix& x) const;
    Vector passForward(const Vector& x) const;
    Vector passForwardWithoutShrinking(const Vector& x) const;
    Gradient passBackwardAndCalcGradient(Vector& u, Vector& z) const;
    void updateAlpha(const Matrix& alpha, double step);
    void updateBeta(const Matrix& beta, double step);
    void updateSigma(const Vector& sigma, double step);

private:
    void ApplyGs(const Vector& angles, Vector& v) const;
    void ReverseApplyGs(const Vector& angles, Vector& v) const;
    Vector CalcVectorD(const Vector& alphas, Vector& u, Vector& z,
                       size_t z_size) const;
    Vector ReverseCalcVectorD(const Vector& betas, Vector& u, Vector& z,
                              size_t z_size) const;
    Random rnd_;
    size_t n_;
    size_t m_;
    size_t min_n_m_;
    Matrix alpha_;
    Matrix beta_;
    Vector sigma_;
};

}  // namespace neural_network
