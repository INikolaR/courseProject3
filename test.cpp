#include "test.h"

#include <unistd.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "Adam.h"
#include "Constant.h"
#include "CustomTypes.h"
#include "GivensLayer.h"
#include "HouseholderLayer.h"
#include "MatrixLayer.h"
#include "Momentum.h"
#include "parser.h"
#include "TestOperations.h"

namespace neural_network {
void test_mnist() {
    DataLoader train = parser::parseMNISTDataset(
        "../train-images-idx3-ubyte/train-images.idx3-ubyte",
        "../train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    DataLoader test = parser::parseMNISTDataset(
        "../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
        "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
    std::vector<int> seeds = {542, 2345, 5674};
    LossFunction loss = LossFunction::Euclid();
    std::vector<std::vector<int>> architectures = {
        {784, 32, 10}, {784, 10, 10}, {784, 2, 10}};
    std::vector<Optimizer> optimizers = {Optimizer{Constant(0.2)},
                                         Optimizer{Momentum(0.1, 0.1)},
                                         Optimizer{Adam(0.1)}};
    size_t batch_size = 6;
    size_t n_of_epochs = 5;
    for (const std::vector<int>& architecture : architectures) {
        for (const Optimizer& optimizer : optimizers) {
            ClassificationReport givens_report =
                getClassificationReportForGivensNets(
                    architecture, seeds, train, test, loss, loss, batch_size,
                    n_of_epochs, optimizer);
            printReport(givens_report);
            ClassificationReport matrix_report =
                getClassificationReportForMatrixNets(
                    architecture, seeds, train, test, loss, loss, batch_size,
                    n_of_epochs, optimizer);
            printReport(matrix_report);
            ClassificationReport householder_report =
                getClassificationReportForHouseholderNets(
                    architecture, seeds, train, test, loss, loss, batch_size,
                    n_of_epochs, optimizer);
            printReport(householder_report);
        }
    }
}

void test_titanic() {
    TrainTestLoaders loaders =
        parser::parseTitanicDataset("../titanic/Titanic-Dataset.csv");
    DataLoader train = loaders.train_dataloader;
    DataLoader test = loaders.test_dataloader;
    std::vector<int> seeds = {542, 2345, 5674};
    LossFunction loss = LossFunction::Euclid();
    std::vector<std::vector<int>> architectures = {
        {19, 32, 2}, {19, 10, 2}, {19, 2, 2}};
    std::vector<Optimizer> optimizers = {Optimizer{Constant(0.2)},
                                         Optimizer{Momentum(0.1, 0.1)},
                                         Optimizer{Adam(0.1)}};
    size_t batch_size = 6;
    size_t n_of_epochs = 5;
    for (const std::vector<int>& architecture : architectures) {
        for (const Optimizer& optimizer : optimizers) {
            BinaryClassificationReport givens_report =
                getBinaryClassificationReportForGivensNets(
                    architecture, seeds, train, test, loss, loss, batch_size,
                    n_of_epochs, optimizer);
            printReport(givens_report);
            BinaryClassificationReport matrix_report =
                getBinaryClassificationReportForMatrixNets(
                    architecture, seeds, train, test, loss, loss, batch_size,
                    n_of_epochs, optimizer);
            printReport(matrix_report);
            BinaryClassificationReport householder_report =
                getBinaryClassificationReportForHouseholderNets(
                    architecture, seeds, train, test, loss, loss, batch_size,
                    n_of_epochs, optimizer);
            printReport(householder_report);
        }
    }
}

void test_boston() {
    TrainTestLoaders loaders =
        parser::parseBostonDataset("../boston/housing.csv");
    DataLoader train = std::move(loaders.train_dataloader);
    DataLoader test = std::move(loaders.test_dataloader);
    std::vector<int> seeds = {542};
    LossFunction loss = LossFunction::Euclid();
    std::vector<std::vector<int>> architectures = {
        {13, 32, 1}, {13, 10, 1}, {13, 2, 1}};
    std::vector<Optimizer> optimizers = {Optimizer{Constant(0.2)},
                                         Optimizer{Momentum(0.1, 0.1)},
                                         Optimizer{Adam(0.1)}};
    size_t batch_size = 6;
    size_t n_of_epochs = 5;
    for (const std::vector<int>& architecture : architectures) {
        for (const Optimizer& optimizer : optimizers) {
            RegressionReport givens_report = getRegressionReportForGivensNets(
                architecture, seeds, train, test, loss, loss, batch_size,
                n_of_epochs, optimizer);
            printReport(givens_report);
            RegressionReport matrix_report = getRegressionReportForMatrixNets(
                architecture, seeds, train, test, loss, loss, batch_size,
                n_of_epochs, optimizer);
            printReport(matrix_report);
            RegressionReport householder_report =
                getRegressionReportForHouseholderNets(
                    architecture, seeds, train, test, loss, loss, batch_size,
                    n_of_epochs, optimizer);
            printReport(householder_report);
        }
    }
}

void run_all_tests() {
    test_boston();
    test_titanic();
    test_mnist();
}

}  // namespace neural_network
