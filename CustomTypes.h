#pragma once
#include <chrono>
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
struct SVD {
    Vector U;
    Vector sigma;
    Vector V;
};
struct CommonMetrics {
    std::string architecture;
    std::string optimizer;
    size_t batch_size;
    double step;
    size_t current_epoch;
    std::chrono::milliseconds::rep epoch_time_ms;
    Vector frobenius_norms;
};
struct ClassificationReport {
    CommonMetrics common_metrics;
    double train_loss;
    double train_accuracy;
    double test_loss;
    double test_accuracy;
};
}  // namespace neural_network
