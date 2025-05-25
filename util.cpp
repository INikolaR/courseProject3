#include "util.h"

#include <cassert>

namespace neural_network {
Eigen::JacobiSVD<Matrix> util::getSVD(In in, Out out,
                                      std::vector<double> weights) {
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

void util::multFirstElemsOfColumnsByVectorElemwise(Matrix& a, const Vector& v) {
    Matrix a_head = a.topRows(v.rows());
    a.topRows(v.rows()) = a_head.array().colwise() * v.array();
}

void util::changeNumberOfRows(Matrix& a, Index new_number_of_rows) {
    Index rows_to_copy =
        new_number_of_rows < a.rows() ? new_number_of_rows : a.rows();
    Matrix resized_a = Matrix::Zero(new_number_of_rows, a.cols());
    resized_a.block(0, 0, rows_to_copy, a.cols()) =
        a.block(0, 0, rows_to_copy, a.cols());
    a = resized_a;
}
}  // namespace neural_network
