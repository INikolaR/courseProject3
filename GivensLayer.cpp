#include "GivensLayer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "VectorOperations.h"

namespace neural_network {
GivensLayer::GivensLayer(In in, Out out, const std::vector<double>& weights)
    : GivensLayer(in, out, getGivensPerfomance(in, out, weights)) {
}

GivensLayer::GivensLayer(In in, Out out, Random& rnd)
    : GivensLayer(in, out, rnd.generateXavier(in, out)) {
}

Matrix GivensLayer::forward(const Matrix& x) const {
    assert(x.rows() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    Matrix temp = Matrix::Ones(x.rows() + 1, x.cols());
    temp.block(0, 0, x.rows(), x.cols()) = std::move(x);
    size_t beta_index = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = n_ - 1; row > col; --row, ++beta_index) {
            GivensRotation(beta_sin_[beta_index], beta_cos_[beta_index], row,
                           temp);
        }
    }
    changeNumberOfRows(temp, m_);
    multFirstElemsOfColumnsByVectorElemwise(temp, sigma_);
    size_t alpha_index = alpha_.size();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, --alpha_index) {
            GivensRotation(-alpha_sin_[alpha_index - 1],
                           alpha_cos_[alpha_index - 1], row, temp);
        }
    }
    return temp;
}

Matrix GivensLayer::forwardOnTrain(const Matrix& x) const {
    assert(x.rows() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    Matrix temp(x.rows() + 1, x.cols());
    temp.block(0, 0, x.rows(), x.cols()) = x;
    temp.row(x.rows()) = Eigen::RowVectorXd::Ones(x.cols());
    size_t beta_index = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = n_ - 1; row > col; --row, ++beta_index) {
            GivensRotation(beta_sin_[beta_index], beta_cos_[beta_index], row,
                           temp);
        }
    }
    changeNumberOfRows(temp, n_ + m_ - min_n_m_);
    multFirstElemsOfColumnsByVectorElemwise(temp, sigma_);
    size_t alpha_index = alpha_.size();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, --alpha_index) {
            GivensRotation(-alpha_sin_[alpha_index - 1],
                           alpha_cos_[alpha_index - 1], row, temp);
        }
    }
    return temp;
}

Matrix GivensLayer::backwardCalcGradient(Matrix& u, const Matrix& x,
                                         Matrix& z) const {
    assert(u.rows() == m_ && "u size should be equal to output size of layer");
    assert(
        z.rows() == m_ + n_ - min_n_m_ &&
        "z size should be equal to max(input size + 1; output size) of layer");
    Matrix gradient = Matrix::Zero(m_ * n_, 1);
    Index gradient_index = 0;
    size_t alpha_index = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = m_ - 1; row > col; --row, ++alpha_index) {
            gradient.col(0)[gradient_index++] =
                (z.row(row).dot(u.row(row - 1)) -
                 z.row(row - 1).dot(u.row(row))) /
                u.cols();
            GivensRotation(alpha_sin_[alpha_index], alpha_cos_[alpha_index],
                           row, u);
            GivensRotation(alpha_sin_[alpha_index], alpha_cos_[alpha_index],
                           row, z);
        }
    }
    multFirstElemsOfColumnsByVectorElemwise(z,
                                            sigma_.array().inverse().matrix());
    for (Index i = 0; i < min_n_m_; ++i) {
        gradient.col(0)[gradient_index++] = u.row(i).dot(z.row(i)) / u.cols();
    }
    multFirstElemsOfColumnsByVectorElemwise(u, sigma_);
    changeNumberOfRows(z, n_);
    changeNumberOfRows(u, n_);
    size_t beta_index = beta_.size();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < n_; ++row, --beta_index) {
            gradient.col(0)[gradient_index++] =
                (z.row(row - 1).dot(u.row(row)) -
                 z.row(row).dot(u.row(row - 1))) /
                u.cols();
            GivensRotation(-beta_sin_[beta_index - 1],
                           beta_cos_[beta_index - 1], row, u);
            GivensRotation(-beta_sin_[beta_index - 1],
                           beta_cos_[beta_index - 1], row, z);
        }
    }
    return gradient;
}

void GivensLayer::update(const Matrix& grad, double step) {
    assert(grad.size() == alpha_.size() + sigma_.size() + beta_.size() &&
           "different shapes of parameter and graient");
    assert(grad.cols() == 1 && "grad should have 1 column");
    alpha_ -= (grad.col(0).segment(0, alpha_.size()) * step).eval();
    sigma_ -= (grad.col(0).segment(alpha_.size(), sigma_.size()) * step).eval();
    beta_ -= (grad.col(0).segment(alpha_.size() + sigma_.size(), beta_.size()) *
              step)
                 .eval();
    alpha_sin_ = alpha_.array().sin();
    alpha_cos_ = alpha_.array().cos();
    beta_sin_ = beta_.array().sin();
    beta_cos_ = beta_.array().cos();
    // sigma weight clipping around 1
    // sigma_ = 2 * 0.01 * (1 / (1 + (-sigma_.array()).exp()) - 0.5) + 1;
}

std::string GivensLayer::describe() const {
    std::stringstream ss;
    ss << "Givens(" << sizeIn() << "," << sizeOut() << ")";
    return ss.str();
}

Index GivensLayer::size() const {
    return alpha_.size() + sigma_.size() + beta_.size();
}

Index GivensLayer::sizeIn() const {
    return n_ - 1;
}

Index GivensLayer::sizeOut() const {
    return m_;
}

GivensLayer::GivensLayer(In in, Out out, const SVD& svd)
    : n_(in + 1),
      m_(out),
      min_n_m_(std::min(n_, m_)),
      alpha_(svd.U),
      sigma_(svd.sigma),
      beta_(svd.V),
      alpha_sin_(std::move(alpha_.array().sin())),
      alpha_cos_(std::move(alpha_.array().cos())),
      beta_sin_(std::move(beta_.array().sin())),
      beta_cos_(std::move(beta_.array().cos())) {
}

MatrixShape GivensLayer::getGradShape() const {
    return MatrixShape{size(), 1};
}

}  // namespace neural_network
