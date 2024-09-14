#include "test.h"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

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

void simple_test_loss(const std::string& test_name, GivensNet& net,
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

double simple_test_loss_accuracy(const std::string& test_name, GivensNet& net,
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
              << " s\n    loss: " << loss
              << "\n    accuracy: " << net.accuracy(test_dataset)
              << "\n    epochs: " << n_of_epochs
              << "\n    batch size: " << batch_size << "\n    step: " << step
              << "\n";
    return loss;
}

void test_echo() {
    std::vector<TrainUnit> dataset{{{1}, {1}}, {{2}, {2}}, {{3}, {3}},
                                   {{4}, {4}}, {{5}, {5}}, {{6}, {6}},
                                   {{7}, {7}}, {{8}, {8}}};
    GivensNet net(1, 1, ActivationFunction::LeakyReLU());
    simple_test_loss("ECHO", net, dataset, LossFunction::Euclid(), dataset,
                     LossFunction::Euclid(), 100, 1, 0.02);
}

void test_sum() {
    std::vector<TrainUnit> dataset{
        {{1, 1}, {2}}, {{1, 2}, {3}}, {{1, 3}, {4}}, {{1, 4}, {5}},
        {{2, 1}, {3}}, {{2, 2}, {4}}, {{2, 3}, {5}}, {{2, 4}, {6}},
        {{3, 1}, {4}}, {{3, 2}, {5}}, {{3, 3}, {6}}, {{3, 4}, {7}},
        {{4, 1}, {5}}, {{4, 2}, {6}}, {{4, 3}, {7}}, {{4, 4}, {8}}};
    GivensNet net(2, 1, ActivationFunction::LeakyReLU());
    simple_test_loss("SUM", net, dataset, LossFunction::Euclid(), dataset,
                     LossFunction::Euclid(), 1000, 16, 0.05);
}

void test_sum_multi_layers() {
    std::vector<TrainUnit> dataset{
        {{1, 1}, {2}}, {{1, 2}, {3}}, {{1, 3}, {4}}, {{1, 4}, {5}},
        {{2, 1}, {3}}, {{2, 2}, {4}}, {{2, 3}, {5}}, {{2, 4}, {6}},
        {{3, 1}, {4}}, {{3, 2}, {5}}, {{3, 3}, {6}}, {{3, 4}, {7}},
        {{4, 1}, {5}}, {{4, 2}, {6}}, {{4, 3}, {7}}, {{4, 4}, {8}}};
    GivensNet net(2, 3, ActivationFunction::LeakyReLU());
    net.AddLayer(1, ActivationFunction::LeakyReLU());
    simple_test_loss("SUM MULTI LAYERS", net, dataset, LossFunction::Euclid(),
                     dataset, LossFunction::Euclid(), 1000, 1, 0.015);
}

void test_mnist() {
    std::vector<TrainUnit> train =
        parseMNISTDataset("../train-images-idx3-ubyte/train-images.idx3-ubyte",
                          "../train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    std::vector<TrainUnit> test =
        parseMNISTDataset("../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
                          "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
    GivensNet net(784, 256, ActivationFunction::Sigmoid());
    net.AddLayer(10, ActivationFunction::Sigmoid());
    double step = 0.2;
    for (size_t i = 0; i < 1000; ++i) {
        double curr_loss = simple_test_loss_accuracy(
            "MNIST", net, train, LossFunction::Euclid(), test,
            LossFunction::Euclid(), 1, 6, step);
    }
}

void test_vector_output() {
    std::vector<TrainUnit> train = {{{}, {0.1, 2}}};
    GivensNet net(0, 2, ActivationFunction::LeakyReLU());
    simple_test_loss("VECTOR OUTPUT", net, train, LossFunction::Euclid(), train,
                     LossFunction::Euclid(), 100, 1, 0.1);
}

void run_all_tests() {
    test_vector_output();
    test_echo();
    test_sum();
    test_sum_multi_layers();
    test_mnist();
}

}  // namespace neural_network
