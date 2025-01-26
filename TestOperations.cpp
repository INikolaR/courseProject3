#include "TestOperations.h"

#include <cassert>
#include <iostream>

namespace neural_network {
void getPrecisionRecallAccuracy(const Net& net,
                                const std::vector<TrainUnit>& dataset,
                                double& precision, double& recall,
                                double& accuracy) {
    assert(!dataset.empty());
    double tp = 0;
    double tn = 0;
    double fp = 0;
    double fn = 0;
    for (const TrainUnit& train_unit : dataset) {
        Vector out = net.predict(train_unit.x);
        assert(out.size() == 2);
        assert(train_unit.y.size() == 2);
        if (out[0] > out[1]) {
            if (train_unit.y[0] > train_unit.y[1]) {
                ++tn;
            } else {
                ++fn;
            }
        } else {
            if (train_unit.y[0] > train_unit.y[1]) {
                ++fp;
            } else {
                ++tp;
            }
        }
    }
    precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    accuracy = (tp + tn) / (tp + tn + fn + fp);
}

double getMSE(const Net& net, const std::vector<TrainUnit>& dataset) {
    assert(!dataset.empty());
    double mse = 0;
    for (const TrainUnit& train_unit : dataset) {
        Vector out = net.predict(train_unit.x);
        assert(out.size() == train_unit.y.size());
        Vector diff = out - train_unit.y;
        mse += dot(diff, diff);
    }
    return mse / static_cast<double>(dataset.size());
}

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

BinaryClassificationReport getBinaryClassificationReport(
    CommonMetrics common_metrics, const Net& net,
    const std::vector<TrainUnit>& train_dataset, const LossFunction& train_loss,
    const std::vector<TrainUnit>& test_dataset, const LossFunction& test_loss) {
    double train_loss_value = net.loss(train_dataset, train_loss);
    double train_accuracy_value = 0;
    double train_precision_value = 0;
    double train_recall_value = 0;
    getPrecisionRecallAccuracy(net, train_dataset, train_precision_value,
                               train_recall_value, train_accuracy_value);
    double test_loss_value = net.loss(test_dataset, test_loss);
    double test_accuracy_value = 0;
    double test_precision_value = 0;
    double test_recall_value = 0;
    getPrecisionRecallAccuracy(net, test_dataset, test_precision_value,
                               test_recall_value, test_accuracy_value);
    return BinaryClassificationReport{
        std::move(common_metrics), train_loss_value,     train_accuracy_value,
        train_precision_value,     train_recall_value,   test_loss_value,
        test_accuracy_value,       test_precision_value, test_recall_value};
}

RegressionReport getRegressionReport(
    CommonMetrics common_metrics, const Net& net,
    const std::vector<TrainUnit>& train_dataset, const LossFunction& train_loss,
    const std::vector<TrainUnit>& test_dataset, const LossFunction& test_loss) {
    double train_loss_value = net.loss(train_dataset, train_loss);
    double train_mse_value = getMSE(net, train_dataset);
    double test_loss_value = net.loss(test_dataset, test_loss);
    double test_mse_value = getMSE(net, test_dataset);
    return RegressionReport{std::move(common_metrics), train_loss_value,
                            train_mse_value, test_loss_value, test_mse_value};
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

void printReport(const ClassificationReport& report) {
    std::cout << "CLASSIFICATION REPORT:\n"
              << stringPerfomance(report.common_metrics)
              << "train:\n       loss: " << report.train_loss
              << "\n       accuracy: " << report.train_accuracy
              << "\ntest:\n       loss: " << report.test_loss
              << "\n       accuracy: " << report.test_accuracy << "\n\n";
}

void printReport(const BinaryClassificationReport& report) {
    std::cout << "BINARY CLASSIFICATION REPORT:\n"
              << stringPerfomance(report.common_metrics)
              << "train:\n       loss: " << report.train_loss
              << "\n       accuracy: " << report.train_accuracy
              << "\n       precision: " << report.train_precision
              << "\n       recall: " << report.train_recall
              << "\ntest:\n       loss: " << report.test_loss
              << "\n       accuracy: " << report.test_accuracy
              << "\n       precision: " << report.test_precision
              << "\n       recall: " << report.test_recall << "\n\n";
}

void printReport(const RegressionReport& report) {
    std::cout << "REGRESSION REPORT:\n"
              << stringPerfomance(report.common_metrics)
              << "train:\n       loss: " << report.train_loss
              << "\n       MSE: " << report.train_mse
              << "\ntest:\n       loss: " << report.test_loss
              << "\n       MSE: " << report.test_mse << "\n\n";
}
}  // namespace neural_network
