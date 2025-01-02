#include "GivensLayer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "VectorOperations.h"

namespace neural_network {

GivensLayer::GivensLayer(const Vector& weights, size_t in, size_t out)
    : GivensLayer(getGivensPerfomance(weights, out, in + 1), in, out) {
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
    size_t index_beta = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = n_ - 1; row > col; --row, ++index_beta) {
            G(beta_[index_beta], row, temp);
        }
    }
    temp.resize(m_);
    vecnmult(temp, sigma_, min_n_m_);
    size_t index_alpha = alpha_.size() - 1;
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, --index_alpha) {
            G(-alpha_[index_alpha], row, temp);
        }
    }
    return temp;
}

Vector GivensLayer::forwardOnTrain(const Vector& x) const {
    assert(x.size() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    assert(!x.empty() && "x should be not empty");
    Vector temp;
    temp.reserve(x.size() + 1);
    std::copy(x.begin(), x.end(), back_inserter(temp));
    temp.emplace_back(1.0);
    size_t index_beta = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = n_ - 1; row > col; --row, ++index_beta) {
            G(beta_[index_beta], row, temp);
        }
    }
    temp.resize(n_ + m_ - min_n_m_);
    vecnmult(temp, sigma_, min_n_m_);
    size_t index_alpha = alpha_.size() - 1;
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, --index_alpha) {
            G(-alpha_[index_alpha], row, temp);
        }
    }
    return temp;
}

Vector GivensLayer::backwardCalcGradient(Vector& u, const Vector& x,
                                         Vector& z) const {
    assert(u.size() == m_ && "u size should be equal to output size of layer");
    assert(
        z.size() == m_ + n_ - min_n_m_ &&
        "z size should be equal to max(input size + 1; output size) of layer");
    Vector gradient;
    gradient.reserve(n_ * m_);
    auto alpha_it = alpha_.begin();
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = m_ - 1; row > col; --row, ++alpha_it) {
            gradient.emplace_back(z[row] * u[row - 1] - z[row - 1] * u[row]);
            RG(-*alpha_it, row, u);
            G(*alpha_it, row, z);
        }
    }
    Vector asigma;
    asigma.reserve(sigma_.size());
    for (size_t i = 0; i < sigma_.size(); ++i) {
        asigma.emplace_back(1 / sigma_[i]);
    }
    vecnmult(z, asigma, min_n_m_);
    Vector d_sigma = elemwisemult(u, z, min_n_m_);
    z.resize(n_);
    gradient.insert(gradient.end(), d_sigma.begin(), d_sigma.end());
    vecnmult(u, sigma_, min_n_m_);
    u.resize(n_);
    auto beta_it = beta_.rbegin();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < n_; ++row, ++beta_it) {
            gradient.emplace_back(z[row - 1] * u[row] - z[row] * u[row - 1]);
            RG(*beta_it, row, u);
            G(-*beta_it, row, z);
        }
    }
    return gradient;
}

void GivensLayer::update(const Vector& grad, double step) {
    assert(grad.size() == sigma_.size() + alpha_.size() + beta_.size() &&
           "different shapes of parameter and graient");
    auto grad_it = grad.begin();
    for (auto alpha_it = alpha_.begin(); alpha_it != alpha_.end(); ++alpha_it) {
        *alpha_it -= *grad_it * step;
        ++grad_it;
    }
    for (auto sigma_it = sigma_.begin(); sigma_it != sigma_.end(); ++sigma_it) {
        *sigma_it -= *grad_it * step;
        ++grad_it;
    }
    for (auto beta_it = beta_.rbegin(); beta_it != beta_.rend(); ++beta_it) {
        *beta_it -= *grad_it * step;
        ++grad_it;
    }
}

GivensLayer::GivensLayer(const SVD& svd, size_t in, size_t out)
    : n_(in + 1),
      m_(out),
      min_n_m_(std::min(n_, m_)),
      alpha_(std::move(svd.U)),
      beta_(std::move(svd.V)),
      sigma_(std::move(svd.sigma)) {
}

}  // namespace neural_network
