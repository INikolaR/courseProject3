#pragma once

#include "CustomTypes.h"
#include "Random.h"

namespace neural_network {
class HouseholderLayer {
public:
    HouseholderLayer(In in, Out out, const std::vector<double>& weights);
    HouseholderLayer(In in, Out out, Random& rnd);

    Index sizeIn() const;
    Index sizeOut() const;
    Matrix forward(const Matrix& x) const;
    Matrix forwardOnTrain(const Matrix& x) const;
    Matrix backwardCalcGradient(Matrix& u, const Matrix& x, Matrix& z) const;
    void update(const Matrix& grad, double step);
    std::string describe() const;
    Index size() const;

private:
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
