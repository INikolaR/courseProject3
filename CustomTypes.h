#pragma once
#include <chrono>
#include <Eigen/Dense>
#include <vector>

namespace neural_network {
using Index = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
enum In : Index;
enum Out : Index;
struct TrainUnit {
    Matrix x;
    Matrix y;
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
    size_t current_epoch;
    std::chrono::milliseconds::rep epoch_time_ms;
    Vector mean_frobenius_norms;
    Vector std_frobenius_norms;
};
struct ClassificationReport {
    CommonMetrics common_metrics;
    double train_loss;
    double train_accuracy;
    double test_loss;
    double test_accuracy;
};
struct BinaryClassificationReport {
    CommonMetrics common_metrics;
    double train_loss;
    double train_accuracy;
    double train_precision;
    double train_recall;
    double test_loss;
    double test_accuracy;
    double test_precision;
    double test_recall;
};
struct RegressionReport {
    CommonMetrics common_metrics;
    double train_loss;
    double train_mse;
    double test_loss;
    double test_mse;
};
}  // namespace neural_network
