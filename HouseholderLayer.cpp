#include "HouseholderLayer.h"

#include <iostream>

#include "Random.h"
#include "VectorOperations.h"

namespace neural_network {
HouseholderLayer::HouseholderLayer(const Vector& weights, size_t in, size_t out)
    : n_(in + 1),
      m_(out),
      min_n_m_(std::min(n_, m_)),
      w_(std::move(getHouseholderPerfomance(weights, out, in + 1))),
      u_(std::vector<Vector::iterator>()),
      sigma_(w_.begin() +
             (min_n_m_ * (min_n_m_ + 1) / 2 + min_n_m_ * (m_ - min_n_m_))),
      v_(std::vector<Vector::iterator>()) {
    u_.reserve(min_n_m_ + 1);
    v_.reserve(min_n_m_ + 1);
    auto u_it = w_.begin();
    for (size_t col = 0; col < min_n_m_; ++col) {
        u_.emplace_back(u_it);
        u_it += m_ - col;
    }
    u_.emplace_back(u_it);
    auto v_it = sigma_ + min_n_m_;
    for (size_t col = 0; col < min_n_m_; ++col) {
        v_.emplace_back(v_it);
        v_it += n_ - col;
    }
    v_.emplace_back(v_it);
}

HouseholderLayer::HouseholderLayer(Random& rnd, size_t in, size_t out)
    : HouseholderLayer(rnd.xavier(in, out), in, out) {
}

size_t HouseholderLayer::sizeIn() const {
    return n_ - 1;
}

size_t HouseholderLayer::sizeOut() const {
    return m_;
}

Vector HouseholderLayer::forward(const Vector& x) const {
    assert(x.size() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    assert(!x.empty() && "x should be not empty");
    Vector temp;
    temp.reserve(x.size() + 1);
    std::copy(x.begin(), x.end(), back_inserter(temp));
    temp.emplace_back(1.0);
    for (auto v_it = v_.begin(); v_it != v_.end() - 1; ++v_it) {
        H(*v_it, *(v_it + 1), temp, n_);
    }
    temp.resize(m_);
    vecnmult(temp.begin(), sigma_, min_n_m_);
    for (auto u_it = u_.rbegin(); u_it != u_.rend() - 1; ++u_it) {
        H(*(u_it + 1), *u_it, temp, m_);
    }
    return temp;
}

Vector HouseholderLayer::forwardOnTrain(const Vector& x) const {
    assert(x.size() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    assert(!x.empty() && "x should be not empty");
    Vector temp;
    temp.reserve(x.size() + 1);
    std::copy(x.begin(), x.end(), back_inserter(temp));
    temp.emplace_back(1.0);
    for (auto v_it = v_.begin(); v_it != v_.end() - 1; ++v_it) {
        H(*v_it, *(v_it + 1), temp, n_);
    }
    temp.resize(m_ + n_ - min_n_m_);
    vecnmult(temp.begin(), sigma_, min_n_m_);
    for (auto u_it = u_.rbegin(); u_it != u_.rend() - 1; ++u_it) {
        H(*(u_it + 1), *u_it, temp, m_);
    }
    return temp;
}

Vector HouseholderLayer::backwardCalcGradient(Vector& u_grad, const Vector& x,
                                              Vector& z) const {
    assert(u_grad.size() == m_ &&
           "u size should be equal to output size of layer");
    assert(
        z.size() == m_ + n_ - min_n_m_ &&
        "z size should be equal to max(input size + 1; output size) of layer");
    Vector gradient;
    gradient.reserve(n_ * m_ + 2 * min_n_m_);
    auto begin_curr_u_it = u_.begin();
    for (size_t col = 0; col < min_n_m_; ++col, ++begin_curr_u_it) {
        H(*begin_curr_u_it, *(begin_curr_u_it + 1), z, m_);
        assert(*(begin_curr_u_it + 1) - (m_ - col) == *begin_curr_u_it);
        auto u_grad_it = u_grad.begin() + col;
        auto z_it = z.begin() + col;
        double u_u_grad_dot = dotn(*begin_curr_u_it, u_grad_it, m_ - col);
        double u_z_dot = dotn(*begin_curr_u_it, z_it, m_ - col);
        for (auto it = *begin_curr_u_it; it != *(begin_curr_u_it + 1);
             ++it, ++u_grad_it, ++z_it) {
            gradient.emplace_back(*u_grad_it * *z_it -
                                  2 * (*u_grad_it * u_z_dot +
                                       u_u_grad_dot * *z_it +
                                       u_z_dot * u_u_grad_dot * 2 * *it));
        }
        H(*begin_curr_u_it, *(begin_curr_u_it + 1), u_grad, m_);
    }
    assert(*begin_curr_u_it == sigma_);
    Vector asigma;
    asigma.reserve(min_n_m_);
    auto sigma_it = sigma_;
    for (size_t i = 0; i < min_n_m_; ++i, ++sigma_it) {
        asigma.emplace_back(1 / *sigma_it);
    }
    vecnmult(z, asigma, min_n_m_);
    Vector d_sigma = elemwisemult(u_grad, z, min_n_m_);
    z.resize(n_);
    gradient.insert(gradient.end(), d_sigma.begin(), d_sigma.end());
    vecnmult(u_grad.begin(), sigma_, min_n_m_);
    u_grad.resize(n_);
    auto end_curr_v_it = v_.rbegin();
    for (size_t col = min_n_m_; col > 0; --col, ++end_curr_v_it) {
        assert(*end_curr_v_it - (n_ - col + 1) == *(end_curr_v_it + 1));
        H(*(end_curr_v_it + 1), *end_curr_v_it, z, n_);
        auto u_grad_it = u_grad.begin() + (col - 1);
        auto z_it = z.begin() + (col - 1);
        double u_u_grad_dot =
            dotn(*(end_curr_v_it + 1), u_grad_it, n_ - (col - 1));
        double u_z_dot = dotn(*(end_curr_v_it + 1), z_it, n_ - (col - 1));
        for (auto it = *(end_curr_v_it + 1); it != *end_curr_v_it;
             ++it, ++u_grad_it, ++z_it) {
            gradient.emplace_back(*u_grad_it * *z_it -
                                  2 * (*u_grad_it * u_z_dot +
                                       u_u_grad_dot * *z_it +
                                       u_z_dot * u_u_grad_dot * 2 * *it));
        }
        H(*(end_curr_v_it + 1), *end_curr_v_it, u_grad, n_);
    }
    assert(*end_curr_v_it == v_.front());
    return gradient;
}

void HouseholderLayer::update(const Vector& grad, double step) {
    assert(grad.size() == w_.size() &&
           "different shapes of parameter and graient");
    auto grad_it = grad.begin();
    size_t size = w_.size();
    for (auto begin_curr_u_it = u_.begin(); begin_curr_u_it != u_.end() - 1;
         ++begin_curr_u_it) {
        double norm = 0;
        for (auto it = *begin_curr_u_it; it != *(begin_curr_u_it + 1);
             ++it, ++grad_it) {
            *it -= *grad_it * step;
            norm += *it * *it;
        }
        norm = sqrt(norm);
        double norm2 = 0;
        for (auto it = *begin_curr_u_it; it != *(begin_curr_u_it + 1); ++it) {
            *it /= norm;
            norm2 += *it * *it;
        }
    }
    for (auto sigma_it = u_.back(); sigma_it != v_.front();
         ++sigma_it, ++grad_it) {
        *sigma_it -= *grad_it * step;
        *sigma_it = 2 * 0.001 * (1 / (1 + exp(-*sigma_it)) - 0.5) + 1;
    }
    for (auto end_curr_v_it = v_.rbegin(); end_curr_v_it != v_.rend() - 1;
         ++end_curr_v_it) {
        double norm = 0;
        for (auto it = *(end_curr_v_it + 1); it != *end_curr_v_it;
             ++it, ++grad_it) {
            *it -= *grad_it * step;
            norm += *it * *it;
        }
        norm = sqrt(norm);
        double norm2 = 0;
        for (auto it = *(end_curr_v_it + 1); it != *end_curr_v_it; ++it) {
            *it /= norm;
            norm2 += *it * *it;
        }
    }
}
}  // namespace neural_network
