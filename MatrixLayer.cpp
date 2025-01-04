#include "MatrixLayer.h"

#include <cassert>

#include "VectorOperations.h"

namespace neural_network {
MatrixLayer::MatrixLayer(const Vector& w, size_t in, size_t out)
    : w_(w), n_(in + 1), m_(out) {
}

size_t MatrixLayer::sizeIn() const {
    return n_ - 1;
}

size_t MatrixLayer::sizeOut() const {
    return m_;
}

Vector MatrixLayer::forward(const Vector& x) const {
    assert(x.size() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    assert(!x.empty() && "x should be not empty");
    Vector res;
    res.reserve(m_);
    size_t counter = 0;
    for (size_t row = 0; row < m_; ++row) {
        double e = 0;
        for (size_t col = 0; col < n_ - 1; ++col) {
            e += x[col] * w_[counter++];
        }
        e += w_[counter++];
        res.emplace_back(e);
    }
    return res;
}

// this is needed for supporting a single interface
Vector MatrixLayer::forwardOnTrain(const Vector& x) const {
    return forward(x);
    // Vector res;
    // res.reserve(std::max(m_, n_));
    // size_t counter = 0;
    // for (size_t row = 0; row < m_; ++row) {
    //     double e = 0;
    //     for (size_t col = 0; col < n_ - 1; ++col) {
    //         e += x[col] * w_[counter++];
    //     }
    //     e += w_[counter++];
    //     res.emplace_back(e);
    // }
    // for (int i = 0; i < std::max(n_, m_) - m_; ++i) {
    //     res.emplace_back(0);
    // }
    // return res;
}

Vector MatrixLayer::backwardCalcGradient(Vector& u, const Vector& x,
                                         Vector& z) const {
    assert(u.size() == m_ && "u size should be equal to output size of layer");
    assert(x.size() == n_ - 1 &&
           "x size should be equal to input size of layer");
    Vector grad;
    grad.reserve(n_ * m_);
    for (size_t i = 0; i < m_; ++i) {
        for (size_t j = 0; j < n_ - 1; ++j) {
            grad.emplace_back(u[i] * x[j]);
        }
        grad.emplace_back(u[i]);
    }
    return grad;
}

void MatrixLayer::update(const Vector& grad, double step) {
    updateVector(w_, grad, step);
}
}  // namespace neural_network
