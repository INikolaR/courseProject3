#include "TestOperations.h"

#include <iostream>

namespace neural_network {
CommonMetrics measure(std::string architecture, std::string optimizer, Net& net,
                      const std::vector<TrainUnit>& train,
                      const LossFunction& loss, size_t batch_size, double step,
                      size_t current_epoch) {
    auto start = std::chrono::system_clock::now();
    Vector norms =
        net.trainOneEpochWithFrobeniusNorms(train, loss, batch_size, step);
    auto end = std::chrono::system_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    return CommonMetrics{std::move(architecture),  std::move(optimizer),
                         std::move(batch_size),    std::move(step),
                         std::move(current_epoch), std::move(time),
                         std::move(norms)};
}

ClassificationReport getClassificationReport(
    CommonMetrics common_metrics, const Net& net,
    const std::vector<TrainUnit>& train_dataset, const LossFunction& train_loss,
    const std::vector<TrainUnit>& test_dataset, const LossFunction& test_loss) {
    double train_loss_value = net.loss(train_dataset, train_loss);
    double train_accuracy_value = net.accuracy(train_dataset);
    double test_loss_value = net.loss(test_dataset, test_loss);
    double test_accuracy_value = net.accuracy(test_dataset);
    return ClassificationReport{std::move(common_metrics), train_loss_value,
                                train_accuracy_value, test_loss_value,
                                test_accuracy_value};
}

std::string stringPerfomance(const CommonMetrics& common_metrics) {
    std::stringstream ss;
    ss << "ARCH: " << common_metrics.architecture
       << "\nOPTIM: " << common_metrics.optimizer
       << "\nbatch_size: " << common_metrics.batch_size
       << "\nstep: " << common_metrics.step
       << "\nepoch_number: " << common_metrics.current_epoch
       << "\ntime: " << common_metrics.epoch_time_ms / 1000 << "."
       << common_metrics.epoch_time_ms % 1000 << "s\nnorms: ";
    for (double norm : common_metrics.frobenius_norms) {
        ss << norm << " ";
    }
    ss << "\n";
    return ss.str();
}

void printReport(const ClassificationReport report) {
    std::cout << "CLASSIFICATION REPORT:\n"
              << stringPerfomance(report.common_metrics)
              << "train loss: " << report.train_loss
              << "\ntrain accuracy: " << report.train_accuracy
              << "\ntest loss: " << report.test_loss
              << "\ntest accuracy: " << report.test_accuracy << "\n\n";
}
}  // namespace neural_network
