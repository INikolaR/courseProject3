#pragma once

#include "CustomTypes.h"
#include "EigenProxyTypes.h"
#include "Random.h"

namespace neural_network {

class GivensLayer {
public:
    using Array = Eigen::ArrayXd;

    GivensLayer(In in, Out out, const std::vector<double>& weights);
    GivensLayer(In in, Out out, Random& rnd);

    Matrix forward(const Matrix& x) const;
    Matrix forwardOnTrain(const Matrix& x) const;
    Matrix backwardCalcGradient(Matrix& u, const Matrix& x, Matrix& z) const;
    void update(const Matrix& grad, double step);
    std::string describe() const;
    Index size() const;
    Index sizeIn() const;
    Index sizeOut() const;
    MatrixShape getGradShape() const;

private:
    static void GivensRotation(double angle, Index row, Matrix& v);
    static void GivensRotation(double sin, double cos, Index row, Matrix& v);
    static Vector getGivensDecompose(Matrix& m);
    static SVD getGivensPerfomance(In in, Out out,
                                   const std::vector<double>& m);

    GivensLayer(In in, Out out, SVD&& svd);

    Index n_;
    Index m_;
    Index min_n_m_;
    Array alpha_;
    Array sigma_;
    Array beta_;
    Array alpha_sin_;
    Array alpha_cos_;
    Array beta_sin_;
    Array beta_cos_;
};

}  // namespace neural_network
