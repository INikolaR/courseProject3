#include "TestOperations.h"

#include <cassert>
#include <chrono>
#include <iostream>

#include "GivensLayer.h"
#include "HouseholderLayer.h"
#include "MatrixLayer.h"

namespace neural_network {
PrecisionRecallAccuracy getPrecisionRecallAccuracy(const Net& net,
                                                   const DataLoader& loader,
                                                   size_t batch_size) {
    double tp = 0;
    double tn = 0;
    double fp = 0;
    double fn = 0;
    for (const TrainUnit& train_unit : loader.getDataset(batch_size)) {
        Matrix out = net.predict(train_unit.x);
        for (Index i = 0; i < out.cols(); ++i) {
            if (out.col(i)[0] > out.col(i)[1]) {
                if (train_unit.y.col(i)[0] > train_unit.y.col(i)[1]) {
                    ++tn;
                } else {
                    ++fn;
                }
            } else {
                if (train_unit.y.col(i)[0] > train_unit.y.col(i)[1]) {
                    ++fp;
                } else {
                    ++tp;
                }
            }
        }
    }
    return PrecisionRecallAccuracy{tp + fp > 0 ? tp / (tp + fp) : 0,
                                   tp + fn > 0 ? tp / (tp + fn) : 0,
                                   (tp + tn) / (tp + tn + fn + fp)};
}

double getLoss(const Net& net, const DataLoader& loader,
               const LossFunction& loss, size_t batch_size) {
    double actual_loss = 0;
    size_t n_of_samples = 0;
    for (const TrainUnit& train_unit : loader.getDataset(batch_size)) {
        Matrix out = net.predict(train_unit.x);
        actual_loss += loss.evaluate0(out, train_unit.y);
        n_of_samples += out.cols();
    }
    return actual_loss / static_cast<double>(n_of_samples);
}

double getAccuracy(const Net& net, const DataLoader& loader,
                   size_t batch_size) {
    double n_of_correct_answers = 0;
    size_t n_of_samples = 0;
    for (const TrainUnit& train_unit : loader.getDataset(batch_size)) {
        Matrix out = net.predict(train_unit.x);
        for (Index col = 0; col < out.cols(); ++col) {
            int maxRowIndexOut, maxRowIndexY;
            out.col(col).maxCoeff(&maxRowIndexOut);
            train_unit.y.col(col).maxCoeff(&maxRowIndexY);
            n_of_correct_answers += maxRowIndexOut == maxRowIndexY;
        }
        n_of_samples += out.cols();
    }
    return n_of_correct_answers / static_cast<double>(n_of_samples);
}

CommonMetrics measure(Net& net, const DataLoader& loader,
                      const LossFunction& loss, size_t n_of_epochs,
                      size_t batch_size, const Optimizer& optimizer) {
    assert(batch_size > 0);
    assert(n_of_epochs > 0);
    Vector norms = Vector::Zero(net.getNumOfLayers());
    auto start = std::chrono::system_clock::now();
    norms += net.fitAndGetMeanGradNorms(loader, loss, n_of_epochs, batch_size,
                                        optimizer);
    auto end = std::chrono::system_clock::now();

    return CommonMetrics{
        std::move(net.describe()),
        std::move(optimizer->describe()),
        std::move(batch_size),
        std::move(n_of_epochs),
        std::move(static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count())),
        std::move(norms / static_cast<long long>(n_of_epochs))};
}

ClassificationReport getClassificationReport(CommonMetrics common_metrics,
                                             const Net& net,
                                             const DataLoader& train_loader,
                                             const LossFunction& train_loss,
                                             const DataLoader& test_loader,
                                             const LossFunction& test_loss,
                                             size_t batch_size) {
    double train_loss_value =
        getLoss(net, train_loader, train_loss, batch_size);
    double train_accuracy_value = getAccuracy(net, train_loader, batch_size);
    double test_loss_value = getLoss(net, test_loader, test_loss, batch_size);
    double test_accuracy_value = getAccuracy(net, test_loader, batch_size);
    return ClassificationReport{std::move(common_metrics), train_loss_value,
                                train_accuracy_value, test_loss_value,
                                test_accuracy_value};
}

BinaryClassificationReport getBinaryClassificationReport(
    CommonMetrics common_metrics, const Net& net,
    const DataLoader& train_loader, const LossFunction& train_loss,
    const DataLoader& test_loader, const LossFunction& test_loss,
    size_t batch_size) {
    double train_loss_value =
        getLoss(net, train_loader, train_loss, batch_size);
    PrecisionRecallAccuracy train_pra =
        getPrecisionRecallAccuracy(net, train_loader, batch_size);
    double test_loss_value = getLoss(net, test_loader, test_loss, batch_size);
    PrecisionRecallAccuracy test_pra =
        getPrecisionRecallAccuracy(net, test_loader, batch_size);
    return BinaryClassificationReport{
        std::move(common_metrics), train_loss_value,   train_pra.precision,
        train_pra.recall,          train_pra.accuracy, test_loss_value,
        test_pra.precision,        test_pra.recall,    test_pra.accuracy};
}

RegressionReport getRegressionReport(CommonMetrics common_metrics,
                                     const Net& net,
                                     const DataLoader& train_loader,
                                     const LossFunction& train_loss,
                                     const DataLoader& test_loader,
                                     const LossFunction& test_loss,
                                     size_t batch_size) {
    double train_loss_value =
        getLoss(net, train_loader, train_loss, batch_size);
    double test_loss_value = getLoss(net, test_loader, test_loss, batch_size);
    return RegressionReport{std::move(common_metrics), train_loss_value,
                            test_loss_value};
}

std::string getStringPerfomance(const CommonMetrics& common_metrics) {
    std::stringstream ss;
    ss << "ARCH: " << common_metrics.architecture
       << "\nOPTIM: " << common_metrics.optimizer
       << "\nbatch_size: " << common_metrics.batch_size
       << "\nepoch_number: " << common_metrics.total_epochs
       << "\ntime: " << common_metrics.epoch_time_ms / 1000 << "."
       << common_metrics.epoch_time_ms % 1000 << "s\nmean norms: ";
    for (Index i = 0; i < common_metrics.mean_frobenius_norms.rows(); ++i) {
        ss << common_metrics.mean_frobenius_norms[i] << " ";
    }
    ss << "\n";
    return ss.str();
}

void printReport(const ClassificationReport& report) {
    std::cout << "CLASSIFICATION REPORT:\n"
              << getStringPerfomance(report.common_metrics)
              << "train:\n       loss: " << report.train_loss
              << "\n       accuracy: " << report.train_accuracy
              << "\ntest:\n       loss: " << report.test_loss
              << "\n       accuracy: " << report.test_accuracy << "\n\n";
}

void printReport(const BinaryClassificationReport& report) {
    std::cout << "BINARY CLASSIFICATION REPORT:\n"
              << getStringPerfomance(report.common_metrics)
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
              << getStringPerfomance(report.common_metrics)
              << "train:\n       loss: " << report.train_loss
              << "\ntest:\n       loss: " << report.test_loss << "\n\n";
}

ClassificationReport getClassificationReportForGivensNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer) {
    assert(architecture.size() > 1);
    CommonMetrics metrics;
    ClassificationReport report;
    int64_t sum_times = 0;
    Vector sum_frobenius_norms = Vector::Zero(architecture.size() - 1);
    double sum_train_loss_value = 0;
    double sum_test_loss_value = 0;
    double sum_train_acc_value = 0;
    double sum_test_acc_value = 0;
    for (int seed : seeds) {
        Random rnd(seed);
        Net net(
            Linear{GivensLayer(In(architecture[0]), Out(architecture[1]), rnd)},
            NonLinear::Sigmoid());
        for (size_t i = 1; i < architecture.size() - 1; ++i) {
            net.addLayer(Linear{GivensLayer(In(architecture[i]),
                                            Out(architecture[i + 1]), rnd)},
                         NonLinear::Sigmoid());
        }
        metrics = measure(net, train_loader, train_loss, n_of_epochs,
                          batch_size, optimizer);
        report = getClassificationReport(metrics, net, train_loader, train_loss,
                                         test_loader, test_loss, batch_size);
        sum_times += metrics.epoch_time_ms;
        sum_frobenius_norms += metrics.mean_frobenius_norms;
        sum_train_loss_value += report.train_loss;
        sum_test_loss_value += report.test_loss;
        sum_train_acc_value += report.train_accuracy;
        sum_test_acc_value += report.test_accuracy;
    }
    return ClassificationReport{
        CommonMetrics{metrics.architecture, metrics.optimizer,
                      metrics.batch_size, metrics.total_epochs,
                      sum_times / seeds.size(),
                      sum_frobenius_norms / seeds.size()},
        sum_train_loss_value / seeds.size(), sum_train_acc_value / seeds.size(),
        sum_test_loss_value / seeds.size(), sum_test_acc_value / seeds.size()};
}

RegressionReport getRegressionReportForGivensNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer) {
    assert(architecture.size() > 1);
    CommonMetrics metrics;
    RegressionReport report;
    int64_t sum_times = 0;
    Vector sum_frobenius_norms = Vector::Zero(architecture.size() - 1);
    double sum_train_loss_value = 0;
    double sum_test_loss_value = 0;
    for (int seed : seeds) {
        Random rnd(seed);
        Net net(
            Linear{GivensLayer(In(architecture[0]), Out(architecture[1]), rnd)},
            NonLinear::Sigmoid());
        for (size_t i = 1; i < architecture.size() - 1; ++i) {
            net.addLayer(Linear{GivensLayer(In(architecture[i]),
                                            Out(architecture[i + 1]), rnd)},
                         NonLinear::Sigmoid());
        }
        metrics = measure(net, train_loader, train_loss, n_of_epochs,
                          batch_size, optimizer);
        report = getRegressionReport(metrics, net, train_loader, train_loss,
                                     test_loader, test_loss, batch_size);
        sum_times += metrics.epoch_time_ms;
        sum_frobenius_norms += metrics.mean_frobenius_norms;
        sum_train_loss_value += report.train_loss;
        sum_test_loss_value += report.test_loss;
    }
    return RegressionReport{
        CommonMetrics{metrics.architecture, metrics.optimizer,
                      metrics.batch_size, metrics.total_epochs,
                      sum_times / seeds.size(),
                      sum_frobenius_norms / seeds.size()},
        sum_train_loss_value / seeds.size(),
        sum_test_loss_value / seeds.size()};
}

BinaryClassificationReport getBinaryClassificationReportForGivensNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer) {
    assert(architecture.size() > 1);
    CommonMetrics metrics;
    BinaryClassificationReport report;
    int64_t sum_times = 0;
    Vector sum_frobenius_norms = Vector::Zero(architecture.size() - 1);
    double sum_train_loss_value = 0;
    double sum_test_loss_value = 0;
    double sum_train_loss = 0;
    double sum_train_precision = 0;
    double sum_train_recall = 0;
    double sum_train_accuracy = 0;
    double sum_test_loss = 0;
    double sum_test_precision = 0;
    double sum_test_recall = 0;
    double sum_test_accuracy = 0;
    for (int seed : seeds) {
        Random rnd(seed);
        Net net(
            Linear{GivensLayer(In(architecture[0]), Out(architecture[1]), rnd)},
            NonLinear::Sigmoid());
        for (size_t i = 1; i < architecture.size() - 1; ++i) {
            net.addLayer(Linear{GivensLayer(In(architecture[i]),
                                            Out(architecture[i + 1]), rnd)},
                         NonLinear::Sigmoid());
        }
        metrics = measure(net, train_loader, train_loss, n_of_epochs,
                          batch_size, optimizer);
        report = getBinaryClassificationReport(metrics, net, train_loader,
                                               train_loss, test_loader,
                                               test_loss, batch_size);
        sum_times += metrics.epoch_time_ms;
        sum_frobenius_norms += metrics.mean_frobenius_norms;
        sum_train_loss_value += report.train_loss;
        sum_test_loss_value += report.test_loss;
        sum_train_precision += report.train_precision;
        sum_train_recall += report.train_recall;
        sum_train_accuracy += report.train_accuracy;
        sum_test_precision += report.test_precision;
        sum_test_recall += report.test_recall;
        sum_test_accuracy += report.test_accuracy;
    }
    return BinaryClassificationReport{
        CommonMetrics{metrics.architecture, metrics.optimizer,
                      metrics.batch_size, metrics.total_epochs,
                      sum_times / seeds.size(),
                      sum_frobenius_norms / seeds.size()},
        sum_train_loss_value / seeds.size(),
        sum_train_precision / seeds.size(),
        sum_train_recall / seeds.size(),
        sum_train_accuracy / seeds.size(),
        sum_test_loss_value / seeds.size(),
        sum_test_precision / seeds.size(),
        sum_test_recall / seeds.size(),
        sum_test_accuracy / seeds.size()};
}

ClassificationReport getClassificationReportForMatrixNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer) {
    assert(architecture.size() > 1);
    CommonMetrics metrics;
    ClassificationReport report;
    int64_t sum_times = 0;
    Vector sum_frobenius_norms = Vector::Zero(architecture.size() - 1);
    double sum_train_loss_value = 0;
    double sum_test_loss_value = 0;
    double sum_train_acc_value = 0;
    double sum_test_acc_value = 0;
    for (int seed : seeds) {
        Random rnd(seed);
        Net net(
            Linear{MatrixLayer(In(architecture[0]), Out(architecture[1]), rnd)},
            NonLinear::Sigmoid());
        for (size_t i = 1; i < architecture.size() - 1; ++i) {
            net.addLayer(Linear{MatrixLayer(In(architecture[i]),
                                            Out(architecture[i + 1]), rnd)},
                         NonLinear::Sigmoid());
        }
        metrics = measure(net, train_loader, train_loss, n_of_epochs,
                          batch_size, optimizer);
        report = getClassificationReport(metrics, net, train_loader, train_loss,
                                         test_loader, test_loss, batch_size);
        sum_times += metrics.epoch_time_ms;
        sum_frobenius_norms += metrics.mean_frobenius_norms;
        sum_train_loss_value += report.train_loss;
        sum_test_loss_value += report.test_loss;
        sum_train_acc_value += report.train_accuracy;
        sum_test_acc_value += report.test_accuracy;
    }
    return ClassificationReport{
        CommonMetrics{metrics.architecture, metrics.optimizer,
                      metrics.batch_size, metrics.total_epochs,
                      sum_times / seeds.size(),
                      sum_frobenius_norms / seeds.size()},
        sum_train_loss_value / seeds.size(), sum_train_acc_value / seeds.size(),
        sum_test_loss_value / seeds.size(), sum_test_acc_value / seeds.size()};
}

RegressionReport getRegressionReportForMatrixNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer) {
    assert(architecture.size() > 1);
    CommonMetrics metrics;
    RegressionReport report;
    int64_t sum_times = 0;
    Vector sum_frobenius_norms = Vector::Zero(architecture.size() - 1);
    double sum_train_loss_value = 0;
    double sum_test_loss_value = 0;
    for (int seed : seeds) {
        Random rnd(seed);
        Net net(
            Linear{MatrixLayer(In(architecture[0]), Out(architecture[1]), rnd)},
            NonLinear::Sigmoid());
        for (size_t i = 1; i < architecture.size() - 1; ++i) {
            net.addLayer(Linear{MatrixLayer(In(architecture[i]),
                                            Out(architecture[i + 1]), rnd)},
                         NonLinear::Sigmoid());
        }
        metrics = measure(net, train_loader, train_loss, n_of_epochs,
                          batch_size, optimizer);
        report = getRegressionReport(metrics, net, train_loader, train_loss,
                                     test_loader, test_loss, batch_size);
        sum_times += metrics.epoch_time_ms;
        sum_frobenius_norms += metrics.mean_frobenius_norms;
        sum_train_loss_value += report.train_loss;
        sum_test_loss_value += report.test_loss;
    }
    return RegressionReport{
        CommonMetrics{metrics.architecture, metrics.optimizer,
                      metrics.batch_size, metrics.total_epochs,
                      sum_times / seeds.size(),
                      sum_frobenius_norms / seeds.size()},
        sum_train_loss_value / seeds.size(),
        sum_test_loss_value / seeds.size()};
}

BinaryClassificationReport getBinaryClassificationReportForMatrixNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer) {
    assert(architecture.size() > 1);
    CommonMetrics metrics;
    BinaryClassificationReport report;
    int64_t sum_times = 0;
    Vector sum_frobenius_norms = Vector::Zero(architecture.size() - 1);
    double sum_train_loss_value = 0;
    double sum_test_loss_value = 0;
    double sum_train_loss = 0;
    double sum_train_precision = 0;
    double sum_train_recall = 0;
    double sum_train_accuracy = 0;
    double sum_test_loss = 0;
    double sum_test_precision = 0;
    double sum_test_recall = 0;
    double sum_test_accuracy = 0;
    for (int seed : seeds) {
        Random rnd(seed);
        Net net(
            Linear{MatrixLayer(In(architecture[0]), Out(architecture[1]), rnd)},
            NonLinear::Sigmoid());
        for (size_t i = 1; i < architecture.size() - 1; ++i) {
            net.addLayer(Linear{MatrixLayer(In(architecture[i]),
                                            Out(architecture[i + 1]), rnd)},
                         NonLinear::Sigmoid());
        }
        metrics = measure(net, train_loader, train_loss, n_of_epochs,
                          batch_size, optimizer);
        report = getBinaryClassificationReport(metrics, net, train_loader,
                                               train_loss, test_loader,
                                               test_loss, batch_size);
        sum_times += metrics.epoch_time_ms;
        sum_frobenius_norms += metrics.mean_frobenius_norms;
        sum_train_loss_value += report.train_loss;
        sum_test_loss_value += report.test_loss;
        sum_train_precision += report.train_precision;
        sum_train_recall += report.train_recall;
        sum_train_accuracy += report.train_accuracy;
        sum_test_precision += report.test_precision;
        sum_test_recall += report.test_recall;
        sum_test_accuracy += report.test_accuracy;
    }
    return BinaryClassificationReport{
        CommonMetrics{metrics.architecture, metrics.optimizer,
                      metrics.batch_size, metrics.total_epochs,
                      sum_times / seeds.size(),
                      sum_frobenius_norms / seeds.size()},
        sum_train_loss_value / seeds.size(),
        sum_train_precision / seeds.size(),
        sum_train_recall / seeds.size(),
        sum_train_accuracy / seeds.size(),
        sum_test_loss_value / seeds.size(),
        sum_test_precision / seeds.size(),
        sum_test_recall / seeds.size(),
        sum_test_accuracy / seeds.size()};
}

ClassificationReport getClassificationReportForHouseholderNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer) {
    assert(architecture.size() > 1);
    CommonMetrics metrics;
    ClassificationReport report;
    int64_t sum_times = 0;
    Vector sum_frobenius_norms = Vector::Zero(architecture.size() - 1);
    double sum_train_loss_value = 0;
    double sum_test_loss_value = 0;
    double sum_train_acc_value = 0;
    double sum_test_acc_value = 0;
    for (int seed : seeds) {
        Random rnd(seed);
        Net net(Linear{HouseholderLayer(In(architecture[0]),
                                        Out(architecture[1]), rnd)},
                NonLinear::Sigmoid());
        for (size_t i = 1; i < architecture.size() - 1; ++i) {
            net.addLayer(
                Linear{HouseholderLayer(In(architecture[i]),
                                        Out(architecture[i + 1]), rnd)},
                NonLinear::Sigmoid());
        }
        metrics = measure(net, train_loader, train_loss, n_of_epochs,
                          batch_size, optimizer);
        report = getClassificationReport(metrics, net, train_loader, train_loss,
                                         test_loader, test_loss, batch_size);
        sum_times += metrics.epoch_time_ms;
        sum_frobenius_norms += metrics.mean_frobenius_norms;
        sum_train_loss_value += report.train_loss;
        sum_test_loss_value += report.test_loss;
        sum_train_acc_value += report.train_accuracy;
        sum_test_acc_value += report.test_accuracy;
    }
    return ClassificationReport{
        CommonMetrics{metrics.architecture, metrics.optimizer,
                      metrics.batch_size, metrics.total_epochs,
                      sum_times / seeds.size(),
                      sum_frobenius_norms / seeds.size()},
        sum_train_loss_value / seeds.size(), sum_train_acc_value / seeds.size(),
        sum_test_loss_value / seeds.size(), sum_test_acc_value / seeds.size()};
}

RegressionReport getRegressionReportForHouseholderNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer) {
    assert(architecture.size() > 1);
    CommonMetrics metrics;
    RegressionReport report;
    int64_t sum_times = 0;
    Vector sum_frobenius_norms = Vector::Zero(architecture.size() - 1);
    double sum_train_loss_value = 0;
    double sum_test_loss_value = 0;
    for (int seed : seeds) {
        Random rnd(seed);
        Net net(Linear{HouseholderLayer(In(architecture[0]),
                                        Out(architecture[1]), rnd)},
                NonLinear::Sigmoid());
        for (size_t i = 1; i < architecture.size() - 1; ++i) {
            net.addLayer(
                Linear{HouseholderLayer(In(architecture[i]),
                                        Out(architecture[i + 1]), rnd)},
                NonLinear::Sigmoid());
        }
        metrics = measure(net, train_loader, train_loss, n_of_epochs,
                          batch_size, optimizer);
        report = getRegressionReport(metrics, net, train_loader, train_loss,
                                     test_loader, test_loss, batch_size);
        sum_times += metrics.epoch_time_ms;
        sum_frobenius_norms += metrics.mean_frobenius_norms;
        sum_train_loss_value += report.train_loss;
        sum_test_loss_value += report.test_loss;
    }
    return RegressionReport{
        CommonMetrics{metrics.architecture, metrics.optimizer,
                      metrics.batch_size, metrics.total_epochs,
                      sum_times / seeds.size(),
                      sum_frobenius_norms / seeds.size()},
        sum_train_loss_value / seeds.size(),
        sum_test_loss_value / seeds.size()};
}

BinaryClassificationReport getBinaryClassificationReportForHouseholderNets(
    const std::vector<int>& architecture, std::vector<int> seeds,
    const DataLoader& train_loader, const DataLoader& test_loader,
    const LossFunction& train_loss, const LossFunction& test_loss,
    size_t batch_size, size_t n_of_epochs, const Optimizer& optimizer) {
    assert(architecture.size() > 1);
    CommonMetrics metrics;
    BinaryClassificationReport report;
    int64_t sum_times = 0;
    Vector sum_frobenius_norms = Vector::Zero(architecture.size() - 1);
    double sum_train_loss_value = 0;
    double sum_test_loss_value = 0;
    double sum_train_loss = 0;
    double sum_train_precision = 0;
    double sum_train_recall = 0;
    double sum_train_accuracy = 0;
    double sum_test_loss = 0;
    double sum_test_precision = 0;
    double sum_test_recall = 0;
    double sum_test_accuracy = 0;
    for (int seed : seeds) {
        Random rnd(seed);
        Net net(Linear{HouseholderLayer(In(architecture[0]),
                                        Out(architecture[1]), rnd)},
                NonLinear::Sigmoid());
        for (size_t i = 1; i < architecture.size() - 1; ++i) {
            net.addLayer(
                Linear{HouseholderLayer(In(architecture[i]),
                                        Out(architecture[i + 1]), rnd)},
                NonLinear::Sigmoid());
        }
        metrics = measure(net, train_loader, train_loss, n_of_epochs,
                          batch_size, optimizer);
        report = getBinaryClassificationReport(metrics, net, train_loader,
                                               train_loss, test_loader,
                                               test_loss, batch_size);
        sum_times += metrics.epoch_time_ms;
        sum_frobenius_norms += metrics.mean_frobenius_norms;
        sum_train_loss_value += report.train_loss;
        sum_test_loss_value += report.test_loss;
        sum_train_precision += report.train_precision;
        sum_train_recall += report.train_recall;
        sum_train_accuracy += report.train_accuracy;
        sum_test_precision += report.test_precision;
        sum_test_recall += report.test_recall;
        sum_test_accuracy += report.test_accuracy;
    }
    return BinaryClassificationReport{
        CommonMetrics{metrics.architecture, metrics.optimizer,
                      metrics.batch_size, metrics.total_epochs,
                      sum_times / seeds.size(),
                      sum_frobenius_norms / seeds.size()},
        sum_train_loss_value / seeds.size(),
        sum_train_precision / seeds.size(),
        sum_train_recall / seeds.size(),
        sum_train_accuracy / seeds.size(),
        sum_test_loss_value / seeds.size(),
        sum_test_precision / seeds.size(),
        sum_test_recall / seeds.size(),
        sum_test_accuracy / seeds.size()};
}
}  // namespace neural_network
