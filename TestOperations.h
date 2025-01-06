#pragma once

#include <vector>

#include "CustomTypes.h"
#include "LossFunction.h"
#include "Net.h"

namespace neural_network {
CommonMetrics measure(std::string architecture, std::string optimizer, Net& net,
                      const std::vector<TrainUnit>& train,
                      const LossFunction& loss, size_t batch_size, double step,
                      size_t current_epoch);
ClassificationReport getClassificationReport(
    CommonMetrics common_metrics, const Net& net,
    const std::vector<TrainUnit>& train_dataset, const LossFunction& train_loss,
    const std::vector<TrainUnit>& test_dataset, const LossFunction& test_loss);
void printReport(const ClassificationReport report);
}  // namespace neural_network
