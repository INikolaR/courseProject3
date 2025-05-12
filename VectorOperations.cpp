#include "VectorOperations.h"

#include <cassert>
#include <cmath>
#include <iostream>

namespace neural_network {
void GivensRotation(double angle, Index row, Matrix& v) {
    assert(row > 0);
    assert(row <= v.rows() - 1);
    Matrix t = v.block(row - 1, 0, 2, v.cols());
    v.block(row - 1, 0, 2, v.cols()) =
        (Eigen::Matrix2d() << cos(angle), -sin(angle), sin(angle), cos(angle))
            .finished() *
        t;
}

void GivensRotation(double sin, double cos, Index row, Matrix& v) {
    assert(row > 0);
    assert(row <= v.rows() - 1);
    Matrix t = v.block(row - 1, 0, 2, v.cols());
    v.block(row - 1, 0, 2, v.cols()) =
        (Eigen::Matrix2d() << cos, -sin, sin, cos).finished() * t;
}

void HouseholderReflection(const Vector& u, Matrix& a) {
    HouseholderReflection(u, a, a.rows());
}

void HouseholderReflection(const Vector& u, Matrix& a, Index a_rows) {
    Matrix scalar_mults =
        u.transpose() * a.topRows(a_rows).bottomRows(u.rows());
    a.topRows(a_rows).bottomRows(u.rows()).noalias() -= 2.0 * u * scalar_mults;
}

Eigen::JacobiSVD<Matrix> getSVD(In in, Out out, std::vector<double> weights) {
    assert(weights.size() == (in + 1) * out && "bad size of weights vector");
    Matrix m(out, in + 1);
    for (Index i = 0; i < out; ++i) {
        for (Index j = 0; j < in + 1; ++j) {
            m(i, j) = weights[i * (in + 1) + j];
        }
    }
    return Eigen::JacobiSVD<Matrix>(m,
                                    Eigen::ComputeThinU | Eigen::ComputeThinV);
}

Vector getGivensDecompose(Matrix& m) {
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

Vector getHouseholderDecompose(Matrix& m) {
    Vector w((m.cols() * (m.cols() + 1)) / 2 +
             (m.rows() - m.cols()) * m.cols());
    Index w_index = 0;
    for (size_t col = 0; col < m.cols(); ++col) {
        Vector c = m.col(col);
        c(col, 0) -= 1;
        c.normalize();
        for (size_t i = col; i < m.rows(); ++i) {
            w[w_index++] = c(i, 0);
        }
        m.applyOnTheLeft(Matrix::Identity(c.size(), c.size()) -
                         2 * c * c.transpose());
    }
    assert(w_index == w.rows());
    return w;
}

SVD getGivensPerfomance(In in, Out out, const std::vector<double>& m) {
    Eigen::JacobiSVD<Matrix> svd = getSVD(in, out, std::move(m));
    Matrix u = svd.matrixU();
    Matrix v = svd.matrixV();
    Vector s = svd.singularValues();
    return {getGivensDecompose(u), s, getGivensDecompose(v)};
}

SVD getHouseholderPerfomance(In in, Out out, const std::vector<double>& m) {
    Eigen::JacobiSVD<Matrix> svd = getSVD(in, out, std::move(m));
    Matrix u = svd.matrixU();
    Matrix v = svd.matrixV();
    Vector s = svd.singularValues();
    // std::cout << u << "\n";
    // std::cout << s << "\n";
    // std::cout << v << "\n";
    return {getHouseholderDecompose(u), s, getHouseholderDecompose(v)};
}

void multFirstElemsOfColumnsByVectorElemwise(Matrix& a, const Vector& v) {
    Matrix a_head = a.topRows(v.rows());
    a.topRows(v.rows()) = a_head.array().colwise() * v.array();
}

void changeNumberOfRows(Matrix& a, Index new_number_of_rows) {
    Index rows_to_copy =
        new_number_of_rows < a.rows() ? new_number_of_rows : a.rows();
    Matrix resized_a = Matrix::Zero(new_number_of_rows, a.cols());
    resized_a.block(0, 0, rows_to_copy, a.cols()) =
        a.block(0, 0, rows_to_copy, a.cols());
    a = resized_a;
}

}  // namespace neural_network
