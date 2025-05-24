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
struct PrecisionRecallAccuracy {
    double precision;
    double recall;
    double accuracy;
};
struct CommonMetrics {
    std::string architecture;
    std::string optimizer;
    size_t batch_size;
    size_t total_epochs;
    size_t epoch_time_ms;
    Vector mean_frobenius_norms;
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
    PrecisionRecallAccuracy train_precision_recall_accuracy;
    double test_loss;
    PrecisionRecallAccuracy test_precision_recall_accuracy;
};
struct RegressionReport {
    CommonMetrics common_metrics;
    double train_loss;
    double test_loss;
};
}  // namespace neural_network
