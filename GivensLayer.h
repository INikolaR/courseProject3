#pragma once
#include "CustomTypes.h"
#include "Random.h"

namespace neural_network {

class GivensLayer {
public:
    GivensLayer(In in, Out out, const std::vector<double>& weights);
    GivensLayer(In in, Out out, Random& rnd);

    Index sizeIn() const;
    Index sizeOut() const;
    Matrix forward(const Matrix& x) const;
    Matrix forwardOnTrain(const Matrix& x) const;
    Matrix backwardCalcGradient(Matrix& u, const Matrix& x, Matrix& z) const;
    void update(const Matrix& grad, double step);
    std::string describe() const;
    Index size() const;

private:
    GivensLayer(In in, Out out, const SVD& svd);

    Index n_;
    Index m_;
    Index min_n_m_;
    Vector alpha_;
    Vector sigma_;
    Vector beta_;
    Vector alpha_sin_;
    Vector alpha_cos_;
    Vector beta_sin_;
    Vector beta_cos_;
};

}  // namespace neural_network
