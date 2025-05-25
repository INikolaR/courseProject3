#pragma once

#include "CustomTypes.h"
#include "EigenProxyTypes.h"
#include "Random.h"

namespace neural_network {
class HouseholderLayer {
public:
    HouseholderLayer(In in, Out out, const std::vector<double>& weights);
    HouseholderLayer(In in, Out out, Random& rnd);

    Matrix forward(const Matrix& x) const;
    Matrix forwardOnTrain(const Matrix& x) const;
    Matrix backwardCalcGradient(Matrix& grad_from_next, const Matrix& x, Matrix& z) const;
    void update(const Matrix& grad, double step);
    std::string describe() const;
    Index size() const;
    Index sizeIn() const;
    Index sizeOut() const;
    MatrixShape getGradShape() const;

private:
    static void HouseholderReflection(const Vector& u, Matrix& a);
    static void HouseholderReflection(const Vector& u, Matrix& a, Index a_rows);
    static Vector getHouseholderDecompose(Matrix& m);
    static SVD getHouseholderPerfomance(In in, Out out,
                                        const std::vector<double>& m);

    HouseholderLayer(In in, Out out, const SVD& svd);

    Index n_;
    Index m_;
    Index min_n_m_;
    Vector u_;
    Vector sigma_;
    Vector v_;
    std::vector<Index> u_starts_;
    std::vector<Index> v_starts_;
};
}  // namespace neural_network
