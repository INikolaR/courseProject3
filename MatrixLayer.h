#pragma once
#include "CustomTypes.h"
#include "Random.h"

namespace neural_network {
class MatrixLayer {
public:
    MatrixLayer(In in, Out out, const std::vector<double>& weights);
    MatrixLayer(In in, Out out, Random& rnd);

    Matrix forward(const Matrix& x) const;
    Matrix forwardOnTrain(const Matrix& x) const;
    Matrix backwardCalcGradient(Matrix& u, const Matrix& x, Matrix& z) const;
    void update(const Matrix& grad, double step);
    std::string describe() const;
    Index size() const;
    Index sizeIn() const;
    Index sizeOut() const;

private:
    Index n_;
    Index m_;
    Matrix w_;
};
}  // namespace neural_network
