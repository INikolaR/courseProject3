#include "test.h"

#include <unistd.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "GivensLayer.h"
#include "HouseholderLayer.h"
#include "MatrixLayer.h"
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
    Vector x;
    x.reserve(image_size);
    for (int j = 0; j < image_size; ++j) {
        x.emplace_back(static_cast<double>(image[j]) / 255.0);
    }
    label_reader.read(reinterpret_cast<char*>(&label), 1);
    Vector y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    y[static_cast<unsigned int>(label)] = 1;
    return TrainUnit{std::move(x), std::move(y)};
}

std::vector<TrainUnit> parseMNISTDataset(
    const std::string& path_to_images_file,
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

    std::vector<TrainUnit> dataset(0);
    for (int i = 0; i < number_of_labels; i++) {
        dataset.push_back(read_mnist_train_unit(file_images, file_labels,
                                                size_of_mnist_image));
    }
    return dataset;
}

void simple_test_loss(const std::string& test_name, Net& net,
                      const std::vector<TrainUnit>& train_dataset,
                      const LossFunction& train_loss,
                      const std::vector<TrainUnit>& test_dataset,
                      const LossFunction& test_loss, size_t n_of_epochs,
                      int batch_size, double step) {
    std::cout << "TEST " << test_name << ":\n";
    auto start = std::chrono::system_clock::now();
    net.fit(train_dataset, train_loss, n_of_epochs, batch_size, step);
    auto end = std::chrono::system_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "    time: " << time / 1000 << "." << time % 1000
              << " s\n    loss: " << net.loss(test_dataset, test_loss)
              << "\n    epochs: " << n_of_epochs
              << "\n    batch size: " << batch_size << "\n    step: " << step
              << "\n";
}

double simple_test_loss_accuracy(const std::string& test_name, Net& net,
                                 const std::vector<TrainUnit>& train_dataset,
                                 const LossFunction& train_loss,
                                 const std::vector<TrainUnit>& test_dataset,
                                 const LossFunction& test_loss,
                                 size_t n_of_epochs, int batch_size,
                                 double step) {
    std::cout << "TEST " << test_name << ":\n";
    auto start = std::chrono::system_clock::now();
    net.fit(train_dataset, train_loss, n_of_epochs, batch_size, step);
    auto end = std::chrono::system_clock::now();
    double loss = net.loss(test_dataset, test_loss);
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "    time: " << time / 1000 << "." << time % 1000
              << " s\n    train loss: " << net.loss(train_dataset, train_loss)
              << "\n    train accuracy: " << net.accuracy(train_dataset)
              << "\n    test loss: " << loss
              << "\n    test accuracy: " << net.accuracy(test_dataset)
              << "\n    epochs: " << n_of_epochs
              << "\n    batch size: " << batch_size << "\n    step: " << step
              << "\n";
    return loss;
}

void test_echo() {
    std::vector<TrainUnit> dataset{{{1}, {1}}, {{2}, {2}}, {{3}, {3}},
                                   {{4}, {4}}, {{5}, {5}}, {{6}, {6}},
                                   {{7}, {7}}, {{8}, {8}}};
    Net net(Linear{GivensLayer({0.5, 0.5}, 1, 1)},
            ActivationFunction::LeakyReLU());
    simple_test_loss("ECHO", net, dataset, LossFunction::Euclid(), dataset,
                     LossFunction::Euclid(), 100, 8, 0.02);
}

void test_sum() {
    std::vector<TrainUnit> dataset{
        {{1, 1}, {2}}, {{1, 2}, {3}}, {{1, 3}, {4}}, {{1, 4}, {5}},
        {{2, 1}, {3}}, {{2, 2}, {4}}, {{2, 3}, {5}}, {{2, 4}, {6}},
        {{3, 1}, {4}}, {{3, 2}, {5}}, {{3, 3}, {6}}, {{3, 4}, {7}},
        {{4, 1}, {5}}, {{4, 2}, {6}}, {{4, 3}, {7}}, {{4, 4}, {8}}};
    Random rnd;
    Net net(Linear{GivensLayer(rnd.kaiming(2, 1), 2, 1)},
            ActivationFunction::LeakyReLU());
    simple_test_loss("SUM", net, dataset, LossFunction::Euclid(), dataset,
                     LossFunction::Euclid(), 100, 16, 0.07);
}

void test_sum_multi_layers() {
    std::vector<TrainUnit> dataset{
        {{1, 1}, {2}}, {{1, 2}, {3}}, {{1, 3}, {4}}, {{1, 4}, {5}},
        {{2, 1}, {3}}, {{2, 2}, {4}}, {{2, 3}, {5}}, {{2, 4}, {6}},
        {{3, 1}, {4}}, {{3, 2}, {5}}, {{3, 3}, {6}}, {{3, 4}, {7}},
        {{4, 1}, {5}}, {{4, 2}, {6}}, {{4, 3}, {7}}, {{4, 4}, {8}}};
    Random rnd;
    Net net(Linear{GivensLayer(rnd.kaiming(2, 3), 2, 3)},
            ActivationFunction::LeakyReLU());
    net.AddLayer(Linear{GivensLayer(rnd.kaiming(3, 1), 3, 1)},
                 ActivationFunction::LeakyReLU());
    simple_test_loss("SUM MULTI LAYERS", net, dataset, LossFunction::Euclid(),
                     dataset, LossFunction::Euclid(), 100, 1, 0.015);
}

void test_square() {
    std::vector<TrainUnit> train = {{{0.1}, {0.01}},
                                    {{0.2}, {0.04}},
                                    {{0.3}, {0.09}},
                                    {{0.4}, {0.16}},
                                    {{0.5}, {0.25}}};
    Random rnd;
    Net net(Linear{GivensLayer(rnd.kaiming(1, 5), 1, 5)},
            ActivationFunction::LeakyReLU());
    net.AddLayer(Linear{GivensLayer(rnd.kaiming(5, 1), 5, 1)},
                 ActivationFunction::Id());
    simple_test_loss("SQUARE", net, train, LossFunction::Euclid(), train,
                     LossFunction::Euclid(), 1000, 10, 0.001);
}

void test_mnist() {
    std::vector<TrainUnit> train =
        parseMNISTDataset("../train-images-idx3-ubyte/train-images.idx3-ubyte",
                          "../train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    std::vector<TrainUnit> test =
        parseMNISTDataset("../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
                          "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
    std::vector<int> seeds = {542,  2345, 5674, 5423, 64,
                              2435, 765,  798,  5234, 23};
    for (int seed : seeds) {
        Random rnd(seed);
        size_t input_size = 784;
        size_t hid_size = 32;
        size_t output_size = 10;
        Vector w0 = rnd.kaiming(input_size, hid_size);
        Net givens_net(Linear{GivensLayer(w0, input_size, hid_size)},
                       ActivationFunction::LeakyReLU());
        Net matrix_net(Linear{MatrixLayer(w0, input_size, hid_size)},
                       ActivationFunction::LeakyReLU());
        Net householder_net(Linear{HouseholderLayer(w0, input_size, hid_size)},
                            ActivationFunction::LeakyReLU());
        Vector w1 = rnd.xavier(hid_size, output_size);
        givens_net.AddLayer(Linear{GivensLayer(w1, hid_size, output_size)},
                            ActivationFunction::Sigmoid());
        matrix_net.AddLayer(Linear{MatrixLayer(w1, hid_size, output_size)},
                            ActivationFunction::Sigmoid());
        householder_net.AddLayer(
            Linear{HouseholderLayer(w1, hid_size, output_size)},
            ActivationFunction::Sigmoid());
        double step = 0.01;
        double curr_loss = simple_test_loss_accuracy(
            "MNIST GIVENS", givens_net, train, LossFunction::Euclid(), test,
            LossFunction::Euclid(), 1, 10, step);
        double curr_loss_2 = simple_test_loss_accuracy(
            "MNIST MATRIX", matrix_net, train, LossFunction::Euclid(), test,
            LossFunction::Euclid(), 1, 10, step);
        double curr_loss_3 = simple_test_loss_accuracy(
            "MNIST HOUSEHOLDER", householder_net, train, LossFunction::Euclid(),
            test, LossFunction::Euclid(), 1, 10, step);
    }
}

void report_mnist() {
    std::vector<TrainUnit> train =
        parseMNISTDataset("../train-images-idx3-ubyte/train-images.idx3-ubyte",
                          "../train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    std::vector<TrainUnit> test =
        parseMNISTDataset("../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
                          "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
    std::vector<int> seeds = {542,  2345, 5674, 5423, 64,
                              2435, 765,  798,  5234, 23};
    for (int seed : seeds) {
        Random rnd(seed);
        size_t input_size = 784;
        size_t hid_size = 32;
        size_t output_size = 10;
        Vector w0 = rnd.kaiming(input_size, hid_size);
        Net givens_net(Linear{GivensLayer(w0, input_size, hid_size)},
                       ActivationFunction::LeakyReLU());
        Net matrix_net(Linear{MatrixLayer(w0, input_size, hid_size)},
                       ActivationFunction::LeakyReLU());
        Net householder_net(Linear{HouseholderLayer(w0, input_size, hid_size)},
                            ActivationFunction::LeakyReLU());
        Vector w1 = rnd.xavier(hid_size, output_size);
        givens_net.AddLayer(Linear{GivensLayer(w1, hid_size, output_size)},
                            ActivationFunction::Sigmoid());
        matrix_net.AddLayer(Linear{MatrixLayer(w1, hid_size, output_size)},
                            ActivationFunction::Sigmoid());
        householder_net.AddLayer(
            Linear{HouseholderLayer(w1, hid_size, output_size)},
            ActivationFunction::Sigmoid());
        size_t batch_size = 8;
        double step = 0.01;
        size_t n_of_epochs = 1;
        std::chrono::milliseconds::rep total_time = 0;
        LossFunction train_loss = LossFunction::Euclid();
        LossFunction test_loss = LossFunction::Euclid();
        for (size_t epoch = 0; epoch < n_of_epochs; ++epoch) {
            CommonMetrics givens_metrics = measure(
                "Givens(784, 32) -> LeakyReLU() -> Givens(32, 10) -> Sigmoid()",
                "ConstantOptimizer", givens_net, train, train_loss, batch_size,
                step, epoch);
            ClassificationReport givens_report = getClassificationReport(
                givens_metrics, givens_net, train, train_loss, test, test_loss);
            printReport(givens_report);

            CommonMetrics matrix_metrics = measure(
                "Matrix(784, 32) -> LeakyReLU() -> Matrix(32, 10) -> Sigmoid()",
                "ConstantOptimizer", matrix_net, train, train_loss, batch_size,
                step, epoch);
            ClassificationReport matrix_report = getClassificationReport(
                matrix_metrics, matrix_net, train, train_loss, test, test_loss);
            printReport(matrix_report);

            CommonMetrics householder_metrics = measure(
                "Householder(784, 32) -> LeakyReLU() -> Householder(32, 10) -> "
                "Sigmoid()",
                "ConstantOptimizer", householder_net, train, train_loss,
                batch_size, step, epoch);
            ClassificationReport householder_report =
                getClassificationReport(householder_metrics, householder_net,
                                        train, train_loss, test, test_loss);
            printReport(householder_report);
        }
    }
}

void run_all_tests() {
    // test_echo();
    // test_sum();
    // test_sum_multi_layers();
    // test_square();
    // test_mnist();
    report_mnist();
}

}  // namespace neural_network
