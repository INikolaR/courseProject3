#include "GivensLayer.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "VectorOperations.h"

namespace neural_network {

GivensLayer::GivensLayer(size_t in, size_t out)
    : rnd_(std::move(Random())),
      n_(in + 1),
      m_(out),
      min_n_m_(std::min(n_, m_)),
      alpha_(std::move(
          rnd_.givensAngleMatrix(std::min(min_n_m_, m_ - 1), m_ - 1))),
      beta_(std::move(
          rnd_.givensAngleMatrix(std::min(min_n_m_, n_ - 1), n_ - 1))),
      sigma_(std::move(rnd_.singularValues(min_n_m_))) {
    size_t r = 0;
}

size_t GivensLayer::sizeIn() const {
    return n_ - 1;
}

size_t GivensLayer::sizeOut() const {
    return m_;
}

Vector GivensLayer::passForward(const Vector& x) const {
    assert(x.size() == n_ &&
           "size of x should be 1 more than input size of layer");
    assert(!x.empty() && "x should be not empty");
    assert(x[x.size() - 1] == 1 && "last elem of x should be equal 1");
    Vector temp(x);
    for (auto it = beta_.begin(); it != beta_.end(); ++it) {
        ApplyGs(*it, temp);
    }
    temp.resize(m_);
    temp *= sigma_;
    for (auto it = alpha_.rbegin(); it != alpha_.rend(); ++it) {
        ReverseApplyGs(*it, temp);
    }
    return temp;
}

Vector GivensLayer::passForwardWithoutShrinking(const Vector& x) const {
    assert(x.size() == n_ &&
           "size of x should be 1 more than input size of layer");
    assert(x[x.size() - 1] == 1 && "last elem of x should be equal 1");
    Vector temp(x);
    for (auto it = beta_.begin(); it != beta_.end(); ++it) {
        ApplyGs(*it, temp);
    }
    temp *= sigma_;
    for (auto it = alpha_.rbegin(); it != alpha_.rend(); ++it) {
        ReverseApplyGs(*it, temp);
    }
    return temp;
}

Gradient GivensLayer::passBackwardAndCalcGradient(Vector& u, Vector& z) const {
    assert(u.size() == m_ && "u size should be equal to output size of layer");
    assert(z.size() == m_ + n_ - min_n_m_ &&
           "z size should be equal to max(input size; output size) of layer");
    Matrix d_alpha;
    d_alpha.reserve(alpha_.size() * (m_ - 1));
    for (auto it = alpha_.begin(); it != alpha_.end(); ++it) {
        d_alpha.emplace_back(CalcVectorD(*it, u, z, m_));
    }
    Vector asigma;
    asigma.reserve(sigma_.size());
    for (size_t i = 0; i < sigma_.size(); ++i) {
        asigma.emplace_back(1 / sigma_[i]);
    }
    z *= asigma;
    z.resize(n_);
    Vector d_sigma = u * z;
    d_sigma.resize(min_n_m_);
    u *= sigma_;
    u.resize(n_);
    Matrix d_beta;
    d_beta.reserve(beta_.size() * (n_ - 1));
    for (auto it = beta_.rbegin(); it != beta_.rend(); ++it) {
        d_beta.emplace_back(ReverseCalcVectorD(*it, u, z, n_));
    }
    return Gradient{d_alpha, d_sigma, d_beta};
}

void GivensLayer::updateAlpha(const Matrix& alpha, double step) {
    assert(alpha.size() == alpha_.size() &&
           "different shapes of parameter and graient");
    for (size_t i = 0; i < alpha_.size(); ++i) {
        assert(alpha[i].size() == alpha_[i].size() &&
               "different shapes of parameter and graient");
        updateVector(alpha_[i], alpha[i], step);
    }
}

void GivensLayer::updateBeta(const Matrix& beta, double step) {
    assert(beta.size() == beta_.size() &&
           "different shapes of parameter and graient");
    for (size_t i = 0; i < beta_.size(); ++i) {
        assert(beta[i].size() == beta_[beta_.size() - i - 1].size() &&
               "different shapes of parameter and graient");
        Vector b(beta[beta_.size() - i - 1]);
        
        std::reverse(b.begin(), b.end());
        updateVector(beta_[i], b, step);
    }
}

void GivensLayer::updateSigma(const Vector& sigma, double step) {
    updateVector(sigma_, sigma, step);
}

Matrix GivensLayer::passForward(const Matrix& x) const {
    Matrix output;
    output.reserve(x.size());
    for (const Vector& v : x) {
        output.emplace_back(passForward(v));
    }
    return output;
}

void GivensLayer::ApplyGs(const Vector& angles, Vector& v) const {
    for (size_t i = 0; i < angles.size(); ++i) {
        G(angles[i], v.size() - 1 - i, v);
    }
}

void GivensLayer::ReverseApplyGs(const Vector& angles, Vector& v) const {
    for (size_t i = angles.size(); i > 0; --i) {
        G(angles[i - 1], i + v.size() - angles.size() - 1, v);
    }
}

Vector GivensLayer::CalcVectorD(const Vector& alphas, Vector& u, Vector& z,
                                size_t z_size) const {
    Vector d;
    d.reserve(alphas.size());
    for (size_t i = 0; i < alphas.size(); ++i) {
        size_t row = z_size - 1 - i;
        d.emplace_back(z[row] * u[row - 1] - z[row - 1] * u[row]);
        RG(alphas[i], row, u);
        G(-alphas[i], row, z);
    }
    return d;
}

Vector GivensLayer::ReverseCalcVectorD(const Vector& betas, Vector& u,
                                       Vector& z, size_t z_size) const {
    Vector d;
    d.reserve(betas.size());
    for (size_t i = betas.size(); i > 0; --i) {
        size_t row = betas.size() - i + 1;
        d.emplace_back(z[row - 1] * u[row] - z[row] * u[row - 1]);
        RG(betas[i - 1], row, u);
        G(-betas[i - 1], row, z);
    }
    return d;
}

}  // namespace neural_network
