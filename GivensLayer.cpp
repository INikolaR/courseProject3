#include "GivensLayer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "VectorOperations.h"

namespace neural_network {

GivensLayer::GivensLayer(size_t in, size_t out)
    : rnd_(std::move(Random())),
      n_(in + 1),
      m_(out),
      min_n_m_(std::min(n_, m_)),
      alpha_(std::move(rnd_.givensAngles(min_n_m_ * (min_n_m_ - 1) / 2 +
                                         min_n_m_ * (m_ - min_n_m_)))),
      beta_(std::move(rnd_.givensAngles(min_n_m_ * (min_n_m_ - 1) / 2 +
                                        min_n_m_ * (n_ - min_n_m_)))),
      sigma_(std::move(rnd_.singularValues(min_n_m_))) {
}

size_t GivensLayer::sizeIn() const {
    return n_ - 1;
}

size_t GivensLayer::sizeOut() const {
    return m_;
}

Vector GivensLayer::forward(const Vector& x) const {
    assert(x.size() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    assert(!x.empty() && "x should be not empty");
    Vector temp;
    temp.reserve(x.size() + 1);
    std::copy(x.begin(), x.end(), back_inserter(temp));
    temp.emplace_back(1.0);
    auto it_beta = beta_.begin();
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = temp.size() - 1; row > col; --row, ++it_beta) {
            G(*it_beta, row, temp);
        }
    }
    temp.resize(m_);
    temp *= sigma_;
    auto it_alpha = alpha_.rbegin();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < temp.size(); ++row, ++it_alpha) {
            G(-*it_alpha, row, temp);
        }
    }
    return temp;
}

Vector GivensLayer::forwardWithoutShrinking(const Vector& x) const {
    assert(x.size() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    assert(!x.empty() && "x should be not empty");
    Vector temp;
    temp.reserve(x.size() + 1);
    std::copy(x.begin(), x.end(), back_inserter(temp));
    temp.emplace_back(1.0);
    auto it_beta = beta_.begin();
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = n_ - 1; row > col; --row, ++it_beta) {
            G(*it_beta, row, temp);
        }
    }
    temp.resize(n_ + m_ - min_n_m_);
    temp *= sigma_;
    auto it_alpha = alpha_.rbegin();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, ++it_alpha) {
            G(-*it_alpha, row, temp);
        }
    }
    return temp;
}

SVD GivensLayer::backwardCalcGradient(Vector& u, Vector& z) const {
    assert(u.size() == m_ && "u size should be equal to output size of layer");
    assert(
        z.size() == m_ + n_ - min_n_m_ &&
        "z size should be equal to max(input size + 1; output size) of layer");
    Vector d_alpha;
    d_alpha.reserve(alpha_.size());
    auto alpha_it = alpha_.begin();
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = m_ - 1; row > col; --row, ++alpha_it) {
            d_alpha.emplace_back(z[row] * u[row - 1] - z[row - 1] * u[row]);
            RG(-*alpha_it, row, u);
            G(*alpha_it, row, z);
        }
    }
    Vector asigma;
    asigma.reserve(sigma_.size());
    for (size_t i = 0; i < sigma_.size(); ++i) {
        asigma.emplace_back(1 / sigma_[i]);
    }
    z *= asigma;
    Vector d_sigma = u * z;
    z.resize(n_);
    d_sigma.resize(min_n_m_);
    u *= sigma_;
    u.resize(n_);
    Vector d_beta;
    d_beta.reserve(beta_.size());
    auto beta_it = beta_.rbegin();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < n_; ++row, ++beta_it) {
            d_beta.emplace_back(z[row - 1] * u[row] - z[row] * u[row - 1]);
            RG(*beta_it, row, u);
            G(-*beta_it, row, z);
        }
    }
    return SVD{d_alpha, d_sigma, d_beta};
}

void GivensLayer::update(const SVD& grad, double step) {
    assert(grad.U.size() == alpha_.size() &&
           "different shapes of parameter and graient");
    assert(grad.V.size() == beta_.size() &&
           "different shapes of parameter and graient");
    assert(grad.sigma.size() == sigma_.size() &&
           "different shapes of parameter and graient");
    updateVector(alpha_, grad.U, step);
    updateVector(beta_, grad.V, step);
    updateVector(sigma_, grad.sigma, step);
}

}  // namespace neural_network
