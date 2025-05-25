#pragma once

#include <vector>

#include "CustomTypes.h"
#include "LossFunction.h"
#include "Net.h"

namespace neural_network {
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
    double train_precision;
    double train_recall;
    double train_accuracy;
    double test_loss;
    double test_precision;
    double test_recall;
    double test_accuracy;
};
struct RegressionReport {
    CommonMetrics common_metrics;
    double train_loss;
    double test_loss;
};

PrecisionRecallAccuracy getPrecisionRecallAccuracy(const Net& net,
                                                   const DataLoader& dataset,
                                                   size_t batch_size);
double getLoss(const Net& net, const DataLoader& loader,
               const LossFunction& loss, size_t batch_size);
double getAccuracy(const Net& net, const DataLoader& loader, size_t batch_size);

CommonMetrics measure(Net& net, const DataLoader& loader,
                      const LossFunction& loss, size_t n_of_epochs,
                      size_t batch_size, const Optimizer& optimizer);
ClassificationReport getClassificationReport(CommonMetrics common_metrics,
                                             const Net& net,
                                             const DataLoader& train_loader,
                                             const LossFunction& train_loss,
                                             const DataLoader& test_loader,
                                             const LossFunction& test_loss,
                                             size_t batch_size);
BinaryClassificationReport getBinaryClassificationReport(
    CommonMetrics common_metrics, const Net& net,
    const DataLoader& train_dataset, const LossFunction& train_loss,
    const DataLoader& test_dataset, const LossFunction& test_loss,
    size_t batch_size);
RegressionReport getRegressionReport(CommonMetrics common_metrics,
                                     const Net& net,
                                     const DataLoader& train_dataset,
                                     const LossFunction& train_loss,
                                     const DataLoader& test_dataset,
                                     const LossFunction& test_loss,
                                     size_t batch_size);

std::string getStringPerfomance(const CommonMetrics& common_metrics);
void printReport(const ClassificationReport& report);
void printReport(const BinaryClassificationReport& report);
void printReport(const RegressionReport& report);

ClassificationReport getClassificationReportForGivensNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer);

RegressionReport getRegressionReportForGivensNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer);

BinaryClassificationReport getBinaryClassificationReportForGivensNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer);

ClassificationReport getClassificationReportForMatrixNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer);

RegressionReport getRegressionReportForMatrixNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer);

BinaryClassificationReport getBinaryClassificationReportForMatrixNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer);

ClassificationReport getClassificationReportForHouseholderNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer);

RegressionReport getRegressionReportForHouseholderNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer);

BinaryClassificationReport getBinaryClassificationReportForHouseholderNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer);
}  // namespace neural_network
