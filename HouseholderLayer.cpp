#include "HouseholderLayer.h"

#include <cassert>
#include <iostream>

#include "VectorOperations.h"

namespace neural_network {
HouseholderLayer::HouseholderLayer(In in, Out out,
                                   const std::vector<double>& weights)
    : HouseholderLayer(in, out, getHouseholderPerfomance(in, out, weights)) {
}

HouseholderLayer::HouseholderLayer(In in, Out out, Random& rnd)
    : HouseholderLayer(in, out, rnd.generateXavier(in, out)) {
}

Matrix HouseholderLayer::forward(const Matrix& x) const {
    assert(x.rows() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    Matrix temp = Matrix::Ones(x.rows() + 1, x.cols());
    temp.block(0, 0, x.rows(), x.cols()) = x;
    for (size_t i = 0; i < v_starts_.size() - 1; ++i) {
        HouseholderReflection(
            v_.segment(v_starts_[i], v_starts_[i + 1] - v_starts_[i]), temp,
            n_);
    }
    changeNumberOfRows(temp, m_);
    multFirstElemsOfColumnsByVectorElemwise(temp, sigma_);
    for (size_t i = u_starts_.size() - 1; i > 0; --i) {
        HouseholderReflection(
            u_.segment(u_starts_[i - 1], u_starts_[i] - u_starts_[i - 1]), temp,
            m_);
    }
    return temp;
}

Matrix HouseholderLayer::forwardOnTrain(const Matrix& x) const {
    assert(x.rows() == n_ - 1 &&
           "size of x should be the same as input size of layer");
    Matrix temp(x.rows() + 1, x.cols());
    temp.block(0, 0, x.rows(), x.cols()) = x;
    temp.row(x.rows()) = Eigen::RowVectorXd::Ones(x.cols());
    for (size_t i = 0; i < v_starts_.size() - 1; ++i) {
        HouseholderReflection(
            v_.segment(v_starts_[i], v_starts_[i + 1] - v_starts_[i]), temp,
            n_);
    }
    changeNumberOfRows(temp, m_ + n_ - min_n_m_);
    multFirstElemsOfColumnsByVectorElemwise(temp, sigma_);
    for (size_t i = u_starts_.size() - 1; i > 0; --i) {
        HouseholderReflection(
            u_.segment(u_starts_[i - 1], u_starts_[i] - u_starts_[i - 1]), temp,
            m_);
    }
    return temp;
}

Matrix HouseholderLayer::backwardCalcGradient(Matrix& grad_from_next,
                                              const Matrix& x,
                                              Matrix& z) const {
    assert(grad_from_next.rows() == m_ &&
           "u size should be equal to output size of layer");
    assert(
        z.rows() == m_ + n_ - min_n_m_ &&
        "z size should be equal to max(input size + 1; output size) of layer");
    Matrix gradient = Matrix::Zero(u_.size() + sigma_.size() + v_.size(), 1);
    for (size_t i = 0; i < min_n_m_; ++i) {
        Vector curr_u =
            u_.segment(u_starts_[i], u_starts_[i + 1] - u_starts_[i]);
        HouseholderReflection(curr_u, z, m_);
        assert(u_starts_[i + 1] - (m_ - i) == u_starts_[i]);
        Matrix grad_from_next_for_curr_u =
            grad_from_next.block(i, 0, m_ - i, grad_from_next.cols());
        Matrix z_for_curr_u = z.block(i, 0, m_ - i, z.cols());
        Vector grad_from_next_dot_curr_u =
            grad_from_next_for_curr_u.transpose() * curr_u;
        Vector z_dot_curr_u = z_for_curr_u.transpose() * curr_u;
        gradient.col(0).segment(u_starts_[i], (m_ - i)) =
            (-2 * (grad_from_next_for_curr_u * z_dot_curr_u +
                   z_for_curr_u * grad_from_next_dot_curr_u))
                    .rowwise()
                    .sum() /
                z.cols() +
            4 * grad_from_next_dot_curr_u.dot(z_dot_curr_u) * curr_u;
        HouseholderReflection(
            u_.segment(u_starts_[i], u_starts_[i + 1] - u_starts_[i]),
            grad_from_next, m_);
    }
    multFirstElemsOfColumnsByVectorElemwise(z,
                                            sigma_.array().inverse().matrix());
    for (Index i = 0; i < min_n_m_; ++i) {
        gradient.col(
            0)[min_n_m_ * (min_n_m_ + 1) / 2 + (m_ - min_n_m_) * min_n_m_ + i] =
            grad_from_next.row(i).dot(z.row(i)) / grad_from_next.cols();
    }
    multFirstElemsOfColumnsByVectorElemwise(grad_from_next, sigma_);
    changeNumberOfRows(z, n_);
    changeNumberOfRows(grad_from_next, n_);
    for (size_t i = min_n_m_; i > 0; --i) {
        assert(v_starts_[i] - (n_ - i + 1) == v_starts_[i - 1]);
        Vector curr_v =
            v_.segment(v_starts_[i - 1], v_starts_[i] - v_starts_[i - 1]);
        HouseholderReflection(curr_v, z, n_);
        Matrix grad_from_next_for_curr_v =
            grad_from_next.block(i - 1, 0, n_ - (i - 1), grad_from_next.cols());
        Matrix z_for_curr_v = z.block(i - 1, 0, n_ - (i - 1), z.cols());
        Vector grad_from_next_dot_curr_v =
            grad_from_next_for_curr_v.transpose() * curr_v;
        Vector z_dot_curr_v = z_for_curr_v.transpose() * curr_v;
        gradient.col(0).segment(min_n_m_ * (min_n_m_ + 1) / 2 +
                                    (m_ - min_n_m_ + 1) * min_n_m_ +
                                    v_starts_[i - 1],
                                (n_ - (i - 1))) =
            (-2 * (grad_from_next_for_curr_v * z_dot_curr_v +
                   z_for_curr_v * grad_from_next_dot_curr_v))
                    .rowwise()
                    .sum() /
                z.cols() +
            4 * grad_from_next_dot_curr_v.dot(z_dot_curr_v) * curr_v;
        HouseholderReflection(curr_v, grad_from_next, n_);
    }
    return gradient;
}

void HouseholderLayer::update(const Matrix& grad, double step) {
    assert(grad.size() == u_.size() + sigma_.size() + v_.size() &&
           "different shapes of parameter and graient");
    assert(grad.cols() == 1 && "grad must have 1 column");

    u_ -= (grad.col(0).segment(0, u_.size()) * step).eval();
    sigma_ -= (grad.col(0).segment(u_.size(), sigma_.size()) * step).eval();
    v_ -= (grad.col(0).segment(u_.size() + sigma_.size(), v_.size()) * step)
              .eval();
    u_ /= u_.norm() + 1e-9;
    v_ /= v_.norm() + 1e-9;
}

std::string HouseholderLayer::describe() const {
    std::stringstream ss;
    ss << "Householder(" << sizeIn() << "," << sizeOut() << ")";
    return ss.str();
}

Index HouseholderLayer::size() const {
    return u_.size() + sigma_.size() + v_.size();
}

Index HouseholderLayer::sizeIn() const {
    return n_ - 1;
}

Index HouseholderLayer::sizeOut() const {
    return m_;
}

HouseholderLayer::HouseholderLayer(In in, Out out, const SVD& svd)
    : n_(in + 1),
      m_(out),
      min_n_m_(std::min(n_, m_)),
      u_(svd.U),
      sigma_(svd.sigma),
      v_(svd.V) {
    u_starts_.reserve(min_n_m_ + 1);
    v_starts_.reserve(min_n_m_ + 1);
    Index curr_u_start = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        u_starts_.emplace_back(curr_u_start);
        curr_u_start += m_ - col;
    }
    u_starts_.emplace_back(curr_u_start);
    Index curr_v_start = 0;
    for (size_t col = 0; col < min_n_m_; ++col) {
        v_starts_.emplace_back(curr_v_start);
        curr_v_start += n_ - col;
    }
    v_starts_.emplace_back(curr_v_start);
}

MatrixShape HouseholderLayer::getGradShape() const {
    return MatrixShape{size(), 1};
}
}  // namespace neural_network
