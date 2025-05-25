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
#include "TestOperations.h"

namespace neural_network {

int reverse_int(int i) {
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return (static_cast<int>(c1) << 24) + (static_cast<int>(c2) << 16) +
           (static_cast<int>(c3) << 8) + c4;
}

int read_reversed_int(std::basic_ifstream<char>& reader) {
    int to_be_read = 0;
    reader.read(reinterpret_cast<char*>(&to_be_read), sizeof(int));
    return reverse_int(to_be_read);
}

TrainUnit read_mnist_train_unit(std::basic_ifstream<char>& image_reader,
                                std::basic_ifstream<char>& label_reader,
                                int image_size) {
    unsigned char image[image_size];
    unsigned char label = 0;
    image_reader.read(reinterpret_cast<char*>(image), image_size);
    Vector x = Vector::Zero(image_size);
    for (int j = 0; j < image_size; ++j) {
        x(j) = (static_cast<double>(image[j]) / 255.0);
    }
    label_reader.read(reinterpret_cast<char*>(&label), 1);
    Vector y = Vector::Zero(10);
    y[static_cast<unsigned int>(label)] = 1;
    return TrainUnit{std::move(x), std::move(y)};
}

DataLoader parseMNISTDataset(const std::string& path_to_images_file,
                             const std::string& path_to_labels_file) {
    std::ifstream file_images(path_to_images_file,
                              std::ios::binary | std::ifstream::in);

    if (!file_images.is_open()) {
        file_images.close();
        throw std::runtime_error("Cannot open image file!");
    }

    int images_magic_number = read_reversed_int(file_images);
    const int expected_images_magic_number = 2051;
    if (images_magic_number != expected_images_magic_number) {
        throw std::runtime_error("Bad MNIST image file!");
    }

    int number_of_images = read_reversed_int(file_images);
    int n_rows = read_reversed_int(file_images);
    int n_cols = read_reversed_int(file_images);
    int size_of_mnist_image = n_rows * n_cols;

    std::ifstream file_labels(path_to_labels_file, std::ios::binary);
    if (!file_labels.is_open()) {
        throw std::runtime_error("Cannot open label file!");
    }

    int labels_magic_number = read_reversed_int(file_labels);
    const int expected_labels_magic_number = 2049;
    if (labels_magic_number != expected_labels_magic_number) {
        throw std::runtime_error("Bad MNIST label file!");
    }

    int number_of_labels = read_reversed_int(file_labels);

    if (number_of_labels != number_of_images) {
        throw std::runtime_error(
            "Different number of rows in images and labels!");
    }

    Matrix x = Matrix::Zero(784, number_of_images);
    Matrix y = Matrix::Zero(10, number_of_labels);
    for (int i = 0; i < number_of_labels; i++) {
        TrainUnit unit = read_mnist_train_unit(file_images, file_labels,
                                               size_of_mnist_image);
        x.col(i) = unit.x;
        y.col(i) = unit.y;
    }
    return DataLoader(x, y);
}

Net::TrainTestLoaders parseTitanicDataset(const std::string& filename) {
    std::ifstream fin(filename, std::ifstream::in);

    if (!fin.is_open()) {
        fin.close();
        throw std::runtime_error("Cannot open file!");
    }

    size_t total_size = 891;
    size_t train_size = 792;
    size_t test_size = total_size - train_size;
    size_t size_in = 19;
    Matrix train_x = Matrix::Zero(size_in, train_size);
    Matrix train_y = Matrix::Zero(2, train_size);
    std::string s;
    getline(fin, s);  // skip headers
    for (size_t i = 0; i < train_size; ++i) {
        int index;
        fin >> index;
        train_y(index, i) = 1;
        for (size_t j = 0; j < size_in; ++j) {
            fin >> train_x(j, i);
        }
    }
    Matrix test_x = Matrix::Zero(size_in, test_size);
    Matrix test_y = Matrix::Zero(2, test_size);
    for (size_t i = 0; i < test_size; ++i) {
        int index;
        fin >> index;
        test_y(index, i) = 1;
        for (size_t j = 0; j < size_in; ++j) {
            fin >> test_x(j, i);
        }
    }
    return {std::move(DataLoader(std::move(train_x), std::move(train_y))),
            std::move(DataLoader(std::move(test_x), std::move(test_y)))};
}

Net::TrainTestLoaders parseBostonDataset(const std::string& filename) {
    std::ifstream fin(filename, std::ifstream::in);

    if (!fin.is_open()) {
        fin.close();
        throw std::runtime_error("Cannot open file!");
    }

    size_t total_size = 506;
    size_t train_size = 455;
    size_t test_size = total_size - train_size;
    size_t size_in = 13;
    Matrix train_x = Matrix::Zero(size_in, train_size);
    Matrix train_y = Matrix::Zero(1, train_size);
    for (size_t i = 0; i < train_size; ++i) {
        for (size_t j = 0; j < size_in; ++j) {
            fin >> train_x(j, i);
        }
        fin >> train_y(0, i);
    }
    Matrix test_x = Matrix::Zero(size_in, test_size);
    Matrix test_y = Matrix::Zero(1, test_size);
    for (size_t i = 0; i < test_size; ++i) {
        for (size_t j = 0; j < size_in; ++j) {
            fin >> test_x(j, i);
        }
        fin >> test_y(0, i);
    }
    return {std::move(DataLoader(std::move(train_x), std::move(train_y))),
            std::move(DataLoader(std::move(test_x), std::move(test_y)))};
}

void report_mnist() {
    DataLoader train =
        parseMNISTDataset("../train-images-idx3-ubyte/train-images.idx3-ubyte",
                          "../train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    DataLoader test =
        parseMNISTDataset("../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
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

void report_titanic() {
    Net::TrainTestLoaders loaders =
        parseTitanicDataset("../titanic/Titanic-Dataset.csv");
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

void report_boston() {
    Net::TrainTestLoaders loaders = parseBostonDataset("../boston/housing.csv");
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

void run_all_reports() {
    report_boston();
    report_titanic();
    report_mnist();
}

}  // namespace neural_network
