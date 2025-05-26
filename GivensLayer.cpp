#include "GivensLayer.h"

#include <cassert>

#include "util.h"

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
    util::changeNumberOfRows(temp, m_);
    util::multFirstElemsOfColumnsByVectorElemwise(temp, sigma_);
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
    util::changeNumberOfRows(temp, n_ + m_ - min_n_m_);
    util::multFirstElemsOfColumnsByVectorElemwise(temp, sigma_);
    size_t alpha_index = alpha_.size();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < m_; ++row, --alpha_index) {
            GivensRotation(-alpha_sin_[alpha_index - 1],
                           alpha_cos_[alpha_index - 1], row, temp);
        }
    }
    return temp;
}

Matrix GivensLayer::backwardCalcGradient(Matrix& grad_from_next,
                                         const Matrix& x, Matrix& z) const {
    assert(grad_from_next.rows() == m_ &&
           "grad_from_next.rows() should be equal to output size of layer");
    assert(x.rows() == n_ - 1 &&
           "x.rows() should be equal to input size of layer");
    assert(
        z.rows() == m_ + n_ - min_n_m_ &&
        "z size should be equal to max(input size + 1; output size) of layer");
    assert(
        grad_from_next.cols() == x.cols() &&
        "batch size (number of cols) of grad_from_next and x should be equal");
    assert(
        grad_from_next.cols() == z.cols() &&
        "batch size (number of cols) of grad_from_next and z should be equal");
    Matrix gradient = Matrix::Zero(m_ * n_, 1);
    Index gradient_index = 0;
    size_t alpha_index = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        for (size_t row = m_ - 1; row > col; --row, ++alpha_index) {
            gradient.col(0)[gradient_index++] =
                (z.row(row).dot(grad_from_next.row(row - 1)) -
                 z.row(row - 1).dot(grad_from_next.row(row))) /
                grad_from_next.cols();
            GivensRotation(alpha_sin_[alpha_index], alpha_cos_[alpha_index],
                           row, grad_from_next);
            GivensRotation(alpha_sin_[alpha_index], alpha_cos_[alpha_index],
                           row, z);
        }
    }
    util::multFirstElemsOfColumnsByVectorElemwise(
        z, sigma_.array().inverse().matrix());
    for (Index i = 0; i < min_n_m_; ++i) {
        gradient.col(0)[gradient_index++] =
            grad_from_next.row(i).dot(z.row(i)) / grad_from_next.cols();
    }
    util::multFirstElemsOfColumnsByVectorElemwise(grad_from_next, sigma_);
    util::changeNumberOfRows(z, n_);
    util::changeNumberOfRows(grad_from_next, n_);
    size_t beta_index = beta_.size();
    for (size_t col = min_n_m_; col > 0; --col) {
        for (size_t row = col; row < n_; ++row, --beta_index) {
            gradient.col(0)[gradient_index++] =
                (z.row(row - 1).dot(grad_from_next.row(row)) -
                 z.row(row).dot(grad_from_next.row(row - 1))) /
                grad_from_next.cols();
            GivensRotation(-beta_sin_[beta_index - 1],
                           beta_cos_[beta_index - 1], row, grad_from_next);
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
    alpha_ -= (grad.col(0).segment(0, alpha_.size()) * step).array().eval();
    sigma_ -= (grad.col(0).segment(alpha_.size(), sigma_.size()) * step)
                  .array()
                  .eval();
    beta_ -= (grad.col(0).segment(alpha_.size() + sigma_.size(), beta_.size()) *
              step)
                 .array()
                 .eval();
    alpha_sin_ = alpha_.sin();
    alpha_cos_ = alpha_.cos();
    beta_sin_ = beta_.sin();
    beta_cos_ = beta_.cos();
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

MatrixShape GivensLayer::getGradShape() const {
    return MatrixShape{size(), 1};
}

void GivensLayer::GivensRotation(double angle, Index row, Matrix& v) {
    assert(row > 0);
    assert(row <= v.rows() - 1);
    Matrix t = v.block(row - 1, 0, 2, v.cols());
    v.block(row - 1, 0, 2, v.cols()) =
        (Eigen::Matrix2d() << cos(angle), -sin(angle), sin(angle), cos(angle))
            .finished() *
        t;
}

void GivensLayer::GivensRotation(double sin, double cos, Index row, Matrix& v) {
    assert(row > 0);
    assert(row <= v.rows() - 1);
    Matrix t = v.block(row - 1, 0, 2, v.cols());
    v.block(row - 1, 0, 2, v.cols()) =
        (Eigen::Matrix2d() << cos, -sin, sin, cos).finished() * t;
}

Vector GivensLayer::getGivensDecompose(Matrix& m) {
    Vector w((m.cols() * (m.cols() - 1)) / 2 +
             (m.rows() - m.cols()) * m.cols());
    Index w_index = 0;
    for (size_t col = 0; col < m.cols(); ++col) {
        for (size_t row = m.rows() - 1; row > col; --row) {
            double angle = atan2(-m(row, col), m(row - 1, col));
            Matrix g{{cos(angle), -sin(angle)}, {sin(angle), cos(angle)}};
            m.block(row - 1, 0, 2, m.cols()).applyOnTheLeft(g);
            w[w_index++] = angle;
        }
    }
    assert(w_index == w.rows());
    return w;
}

SVD GivensLayer::getGivensPerfomance(In in, Out out,
                                     const std::vector<double>& m) {
    Eigen::JacobiSVD<Matrix> svd = util::getSVD(in, out, std::move(m));
    Matrix u = svd.matrixU();
    Matrix v = svd.matrixV();
    Vector s = svd.singularValues();
    return {getGivensDecompose(u), s, getGivensDecompose(v)};
}

GivensLayer::GivensLayer(In in, Out out, SVD&& svd)
    : n_(in + 1),
      m_(out),
      min_n_m_(std::min(n_, m_)),
      alpha_(std::move(svd.U)),
      sigma_(std::move(svd.sigma)),
      beta_(std::move(svd.V)),
      alpha_sin_(alpha_.array().sin()),
      alpha_cos_(alpha_.array().cos()),
      beta_sin_(beta_.array().sin()),
      beta_cos_(beta_.array().cos()) {
}
}  // namespace neural_network
