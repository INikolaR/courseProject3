#include "GivensLayer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "VectorOperations.h"

namespace neural_network {

GivensLayer::GivensLayer(const Vector& weights, size_t in, size_t out)
    : n_(in + 1),
      m_(out),
      min_n_m_(std::min(n_, m_)),
      w_(std::move(getGivensPerfomance(weights, out, in + 1))),
      alpha_(w_.begin()),
      sigma_(alpha_ +
             (min_n_m_ * (min_n_m_ - 1) / 2 + (m_ - min_n_m_) * min_n_m_)),
      beta_(sigma_ + min_n_m_) {
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
    auto beta_it = beta_;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = n_ - 1; row > col; --row, ++beta_it) {
            G(*beta_it, row, temp);
        }
    }
    temp.resize(m_);
    vecnmult(temp.begin(), sigma_, min_n_m_);
    auto alpha_it = sigma_;  // end of alpha_
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, --alpha_it) {
            G(-*(alpha_it - 1), row, temp);
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
    auto beta_it = beta_;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = n_ - 1; row > col; --row, ++beta_it) {
            G(*beta_it, row, temp);
        }
    }
    temp.resize(n_ + m_ - min_n_m_);
    vecnmult(temp.begin(), sigma_, min_n_m_);
    auto alpha_it = sigma_;  // end of alpha_
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, --alpha_it) {
            G(-*(alpha_it - 1), row, temp);
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
    auto alpha_it = alpha_;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = m_ - 1; row > col; --row, ++alpha_it) {
            gradient.emplace_back(z[row] * u[row - 1] - z[row - 1] * u[row]);
            RG(-*alpha_it, row, u);
            G(*alpha_it, row, z);
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
    vecnmult(u.begin(), sigma_, min_n_m_);
    u.resize(n_);
    auto beta_it = w_.rbegin();
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
    assert(grad.size() == w_.size() &&
           "different shapes of parameter and graient");
    auto grad_it = grad.begin();
    for (auto alpha_it = alpha_; alpha_it != sigma_; ++alpha_it) {
        *alpha_it -= *grad_it * step;
        ++grad_it;
    }
    for (auto sigma_it = sigma_; sigma_it != beta_; ++sigma_it) {
        *sigma_it -= *grad_it * step;
        ++grad_it;
        // *sigma_it = 2 * 0.001 * (1 / (1 + exp(-*sigma_it)) - 0.5) + 1;
    }
    for (auto beta_it = w_.end(); beta_it != beta_; --beta_it) {
        *(beta_it - 1) -= *grad_it * step;
        ++grad_it;
    }
}

}  // namespace neural_network
