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

GivensLayer::GivensLayer(Random& rnd, size_t in, size_t out)
    : GivensLayer(rnd.xavier(in, out), in, out) {
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
    size_t beta_index = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = n_ - 1; row > col; --row, ++beta_index) {
            G(beta_sin_[beta_index], beta_cos_[beta_index], row, temp);
        }
    }
    temp.resize(m_);
    vecnmult(temp, sigma_, min_n_m_);
    size_t alpha_index = alpha_.size();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, --alpha_index) {
            G(-alpha_sin_[alpha_index - 1], alpha_cos_[alpha_index - 1], row,
              temp);
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
    size_t beta_index = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = n_ - 1; row > col; --row, ++beta_index) {
            G(beta_sin_[beta_index], beta_cos_[beta_index], row, temp);
        }
    }
    temp.resize(m_ + n_ - min_n_m_);
    vecnmult(temp, sigma_, min_n_m_);
    size_t alpha_index = alpha_.size();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, --alpha_index) {
            G(-alpha_sin_[alpha_index - 1], alpha_cos_[alpha_index - 1], row,
              temp);
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
    size_t alpha_index = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = m_ - 1; row > col; --row, ++alpha_index) {
            gradient.emplace_back(z[row] * u[row - 1] - z[row - 1] * u[row]);
            G(alpha_sin_[alpha_index], alpha_cos_[alpha_index], row, u);
            G(alpha_sin_[alpha_index], alpha_cos_[alpha_index], row, z);
        }
    }
    Vector asigma;
    asigma.reserve(min_n_m_);
    for (size_t i = 0; i < min_n_m_; ++i) {
        asigma.emplace_back(1 / sigma_[i]);
    }
    vecnmult(z, asigma, min_n_m_);
    Vector d_sigma = elemwisemult(u, z, min_n_m_);
    z.resize(n_);
    gradient.insert(gradient.end(), d_sigma.begin(), d_sigma.end());
    vecnmult(u, sigma_, min_n_m_);
    u.resize(n_);
    size_t beta_index = beta_.size();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < n_; ++row, --beta_index) {
            gradient.emplace_back(z[row - 1] * u[row] - z[row] * u[row - 1]);
            G(-beta_sin_[beta_index - 1], beta_cos_[beta_index - 1], row, u);
            G(-beta_sin_[beta_index - 1], beta_cos_[beta_index - 1], row, z);
        }
    }
    return gradient;
}

void GivensLayer::update(const Vector& grad, double step) {
    assert(grad.size() == alpha_.size() + sigma_.size() + beta_.size() &&
           "different shapes of parameter and graient");
    size_t grad_index = 0;
    for (size_t alpha_index = 0; alpha_index != alpha_.size(); ++alpha_index) {
        alpha_[alpha_index] -= grad[grad_index] * step;
        ++grad_index;
        alpha_sin_[alpha_index] = sin(alpha_[alpha_index]);
        alpha_cos_[alpha_index] = cos(alpha_[alpha_index]);
    }
    for (size_t sigma_index = 0; sigma_index != sigma_.size(); ++sigma_index) {
        sigma_[sigma_index] -= grad[grad_index] * step;
        ++grad_index;
        // sigma_[sigma_index] = 2 * 0.001 * (1 / (1 +
        // exp(-sigma_[sigma_index])) - 0.5) + 1;
    }
    for (size_t beta_index = beta_.size(); beta_index != 0; --beta_index) {
        beta_[beta_index - 1] -= grad[grad_index] * step;
        ++grad_index;
        beta_sin_[beta_index - 1] = sin(beta_[beta_index - 1]);
        beta_cos_[beta_index - 1] = cos(beta_[beta_index - 1]);
    }
}

GivensLayer::GivensLayer(const SVD& svd, size_t in, size_t out)
    : n_(in + 1),
      m_(out),
      min_n_m_(std::min(n_, m_)),
      alpha_(std::move(svd.U)),
      sigma_(std::move(svd.sigma)),
      beta_(std::move(svd.V)),
      alpha_sin_(std::move(sinus(alpha_))),
      alpha_cos_(std::move(cosinus(alpha_))),
      beta_sin_(std::move(sinus(beta_))),
      beta_cos_(std::move(cosinus(beta_))) {
}

}  // namespace neural_network
