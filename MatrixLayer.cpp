#include "MatrixLayer.h"

#include <cassert>
#include <iostream>

#include "VectorOperations.h"

namespace neural_network {
MatrixLayer::MatrixLayer(In in, Out out, const std::vector<double>& w)
    : n_(in + 1), m_(out) {
    w_ = Matrix(out, in + 1);
    for (Index i = 0; i < out; ++i) {
        for (Index j = 0; j < in + 1; ++j) {
            w_(i, j) = w[i * (in + 1) + j];
        }
    }
}

MatrixLayer::MatrixLayer(In in, Out out, Random& rnd)
    : MatrixLayer(in, out, rnd.generateXavier(in, out)) {
}

Matrix MatrixLayer::forward(const Matrix& x) const {
    assert(x.rows() == n_ - 1 &&
           "x.rows() should be the same as input size of layer");
    return (w_.block(0, 0, w_.rows(), w_.cols() - 1) * x).colwise() +
           w_.col(w_.cols() - 1);
}

Matrix MatrixLayer::forwardOnTrain(const Matrix& x) const {
    return forward(x);
}

Matrix MatrixLayer::backwardCalcGradient(Matrix& u, const Matrix& x,
                                         Matrix& z) const {
    assert(u.rows() == m_ && "u size should be equal to output size of layer");
    assert(x.rows() == n_ - 1 &&
           "x size should be equal to input size of layer");
    assert(u.cols() == x.cols() &&
           "batch size (number of cols) should be equal");
    assert(u.cols() == z.cols() &&
           "batch size (number of cols) should be equal");
    Matrix grad = Matrix::Zero(m_, n_);
    grad.block(0, 0, m_, n_ - 1) = u * x.transpose();
    grad.block(0, n_ - 1, m_, 1) = u.rowwise().sum();
    u = (w_.transpose() * u).eval();
    return grad;
}

void MatrixLayer::update(const Matrix& grad, double step) {
    w_ -= grad * step;
}

std::string MatrixLayer::describe() const {
    std::stringstream ss;
    ss << "Matrix(" << sizeIn() << "," << sizeOut() << ")";
    return ss.str();
}

Index MatrixLayer::size() const {
    return w_.size();
}

Index MatrixLayer::sizeIn() const {
    return n_ - 1;
}

Index MatrixLayer::sizeOut() const {
    return m_;
}

MatrixShape MatrixLayer::getGradShape() const {
    return MatrixShape{m_, n_};
}
}  // namespace neural_network
