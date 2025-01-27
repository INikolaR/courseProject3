#pragma once

#include <vector>

#include "CustomTypes.h"
#include "LossFunction.h"
#include "Net.h"

namespace neural_network {
void getPrecisionRecallAccuracy(const Net& net,
                                const std::vector<TrainUnit>& dataset,
                                double& precision, double& recall,
                                double& accuracy);
double getMSE(const Net& net, const std::vector<TrainUnit>& dataset);
CommonMetrics measure(Net& net, const std::vector<TrainUnit>& train,
                      const LossFunction& loss, size_t batch_size,
                      Optimizer& optimizer, size_t current_epoch);
ClassificationReport getClassificationReport(
    CommonMetrics common_metrics, const Net& net,
    const std::vector<TrainUnit>& train_dataset, const LossFunction& train_loss,
    const std::vector<TrainUnit>& test_dataset, const LossFunction& test_loss);
BinaryClassificationReport getBinaryClassificationReport(
    CommonMetrics common_metrics, const Net& net,
    const std::vector<TrainUnit>& train_dataset, const LossFunction& train_loss,
    const std::vector<TrainUnit>& test_dataset, const LossFunction& test_loss);
RegressionReport getRegressionReport(
    CommonMetrics common_metrics, const Net& net,
    const std::vector<TrainUnit>& train_dataset, const LossFunction& train_loss,
    const std::vector<TrainUnit>& test_dataset, const LossFunction& test_loss);
std::string stringPerfomance(const CommonMetrics& common_metrics);
void printReport(const ClassificationReport& report);
void printReport(const BinaryClassificationReport& report);
void printReport(const RegressionReport& report);
}  // namespace neural_network
