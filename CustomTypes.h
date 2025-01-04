#pragma once
#include <Eigen/Dense>
#include <vector>

namespace neural_network {
using Vector = std::vector<double>;
using EMatrix = Eigen::MatrixXd;
using EVector = Eigen::VectorXd;
struct TrainUnit {
    Vector x;
    Vector y;
};
}  // namespace neural_network
