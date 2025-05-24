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
#include "DoubleCsvField.h"
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

    number_of_images /= 10;
    number_of_labels /= 10;
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

// void parseTitanicDataset(const std::string& filename,
//                          std::vector<TrainUnit>& train_dataset,
//                          std::vector<TrainUnit>& test_dataset) {
//     std::ifstream fin(filename, std::ifstream::in);

//     if (!fin.is_open()) {
//         fin.close();
//         throw std::runtime_error("Cannot open file!");
//     }
//     std::string s;
//     train_dataset.clear();
//     train_dataset.reserve(99 * 8);
//     test_dataset.clear();
//     test_dataset.reserve(99);
//     size_t n_pclass[3] = {0, 0, 0};
//     size_t n_sex[2] = {0, 0};
//     size_t n_embarked[3] = {0, 0, 0};
//     DoubleCsvField age;
//     DoubleCsvField sib_sp;
//     DoubleCsvField par_ch;
//     DoubleCsvField fare;
//     getline(fin, s, '\n');  // headers
//     for (size_t i = 0; i < 99 * 8; ++i) {
//         getline(fin, s, ',');  // passengerId
//         getline(fin, s, ',');  // survived
//         assert(s == "0" || s == "1");
//         getline(fin, s, ',');  // PClass
//         assert(s == "1" || s == "2" || s == "3" || s.empty());
//         if (!s.empty()) {
//             ++n_pclass[s[0] - '1'];
//         }
//         getline(fin, s, ',');  // first part of name
//         getline(fin, s, ',');  // second part of name
//         getline(fin, s, ',');  // Sex
//         assert(s == "male" || s == "female" || s.empty());
//         if (!s.empty()) {
//             ++n_sex[s == "male"];
//         }
//         getline(fin, s, ',');  // Age
//         age.processString(s);
//         getline(fin, s, ',');  // SibSp
//         sib_sp.processString(s);
//         getline(fin, s, ',');  // Parch
//         par_ch.processString(s);
//         getline(fin, s, ',');  // Ticket
//         getline(fin, s, ',');  // Fare
//         fare.processString(s);
//         getline(fin, s, ',');   // Cabin
//         getline(fin, s, '\n');  // Embarked
//         assert(s == "S" || s == "C" || s == "Q" || s.empty());
//         if (!s.empty()) {
//             ++n_embarked[s == "S" ? 0 : (s == "C" ? 1 : 2)];
//         }
//     }
//     size_t max_n_pclass =
//         std::max(n_pclass[0], std::max(n_pclass[1], n_pclass[2]));
//     std::string pclass_filler = max_n_pclass == n_pclass[0]   ? "1"
//                                 : max_n_pclass == n_pclass[1] ? "2"
//                                                               : "3";
//     std::string sex_filler =
//         std::max(n_sex[0], n_sex[1]) == n_sex[0] ? "male" : "female";
//     size_t max_n_embarked =
//         std::max(std::max(n_embarked[0], n_embarked[1]), n_embarked[2]);
//     std::string embarked_filler = max_n_embarked == n_pclass[0]   ? "S"
//                                   : max_n_embarked == n_pclass[1] ? "C"
//                                                                   : "Q";
//     fin.seekg(0);
//     getline(fin, s, '\n');  // headers
//     for (size_t i = 0; i < 891; ++i) {
//         Vector x;
//         x.reserve(9);

//         getline(fin, s, ',');  // passengerId
//         getline(fin, s, ',');  // survived
//         assert(s == "0" || s == "1");
//         Vector y{0, 0};
//         y[s[0] - '0'] = 1;
//         getline(fin, s, ',');  // PClass
//         if (s.empty()) {
//             s = pclass_filler;
//         }
//         assert(s == "1" || s == "2" || s == "3");
//         x.emplace_back(s == "1");
//         x.emplace_back(s == "2");
//         getline(fin, s, ',');  // first part of name
//         getline(fin, s, ',');  // second part of name
//         getline(fin, s, ',');  // Sex
//         if (s.empty()) {
//             s = sex_filler;
//         }
//         assert(s == "male" || s == "female");
//         x.emplace_back(s == "male");
//         getline(fin, s, ',');  // Age
//         x.emplace_back(age.evaluate(s));
//         getline(fin, s, ',');  // SibSp
//         x.emplace_back(round(sib_sp.evaluate(s)));
//         getline(fin, s, ',');  // Parch
//         x.emplace_back(round(par_ch.evaluate(s)));
//         getline(fin, s, ',');  // Ticket
//         getline(fin, s, ',');  // Fare
//         x.emplace_back(fare.evaluate(s));
//         getline(fin, s, ',');   // Cabin
//         getline(fin, s, '\n');  // Embarked
//         if (s.empty()) {
//             s = embarked_filler;
//         }
//         assert(s == "S" || s == "C" || s == "Q");
//         x.emplace_back(s == "S");
//         x.emplace_back(s == "C");
//         if (i < 99 * 8) {
//             train_dataset.emplace_back(TrainUnit{std::move(x), std::move(y)});
//         } else {
//             test_dataset.emplace_back(TrainUnit{std::move(x), std::move(y)});
//         }
//     }
// }

Net::TrainTestLoaders parseBostonDataset(const std::string& filename) {
    std::ifstream fin(filename, std::ifstream::in);

    if (!fin.is_open()) {
        fin.close();
        throw std::runtime_error("Cannot open file!");
    }

    size_t total_size = 506;
    size_t train_size = 455;
    size_t size_in = 13;
    double value;
    std::vector<double> minimum(size_in, std::numeric_limits<double>::max());
    std::vector<double> maximum(size_in, std::numeric_limits<double>::min());
    for (size_t i = 0; i < train_size; ++i) {
        for (size_t j = 0; j < 13; ++j) {
            fin >> value;
            minimum[j] = std::min(minimum[j], value);
            maximum[j] = std::max(maximum[j], value);
        }
        fin >> value;
    }
    fin.close();
    fin = std::ifstream(filename, std::ifstream::in);
    if (!fin.is_open()) {
        fin.close();
        throw std::runtime_error("Cannot open file!");
    }
    Matrix train_x = Matrix::Zero(size_in, train_size);
    Matrix train_y = Matrix::Zero(1, train_size);
    for (size_t i = 0; i < train_size; ++i) {
        Vector y;
        for (size_t j = 0; j < 13; ++j) {
            fin >> value;
            train_x(j, i) =
                (maximum[j] == minimum[j]
                     ? 0
                     : (value - minimum[j]) / (maximum[j] - minimum[j]));
        }
        fin >> train_y(1, i);
    }
    Matrix test_x = Matrix::Zero(size_in, train_size);
    Matrix test_y = Matrix::Zero(1, train_size);
    for (size_t i = train_size; i < 506; ++i) {
        for (size_t j = 0; j < 13; ++j) {
            fin >> value;
            test_x(j, i - train_size) =
                (maximum[j] == minimum[j]
                     ? 0
                     : (value - minimum[j]) / (maximum[j] - minimum[j]));
        }
        fin >> test_y(1, i);
    }
    return {std::move(DataLoader(std::move(train_x), std::move(train_y))),
            std::move(DataLoader(std::move(test_x), std::move(test_y)))};
}

// void simple_test_loss(const std::string& test_name, Net& net,
//                       const std::vector<TrainUnit>& train_dataset,
//                       const LossFunction& train_loss,
//                       const std::vector<TrainUnit>& test_dataset,
//                       const LossFunction& test_loss, size_t n_of_epochs,
//                       int batch_size, Optimizer& optimizer) {
//     std::cout << "TEST " << test_name << ":\n";
//     auto start = std::chrono::system_clock::now();
//     net.fit(train_dataset, train_loss, n_of_epochs, batch_size, optimizer);
//     auto end = std::chrono::system_clock::now();
//     auto time =
//         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
//             .count();
//     std::cout << "    time: " << time / 1000 << "." << time % 1000
//               << " s\n    loss: " << net.loss(test_dataset, test_loss)
//               << "\n    epochs: " << n_of_epochs
//               << "\n    batch size: " << batch_size
//               << "\n    optimizer: " << optimizer->describe() << "\n";
// }

// double simple_test_loss_accuracy(const std::string& test_name, Net& net,
//                                  const std::vector<TrainUnit>& train_dataset,
//                                  const LossFunction& train_loss,
//                                  const std::vector<TrainUnit>& test_dataset,
//                                  const LossFunction& test_loss,
//                                  size_t n_of_epochs, int batch_size,
//                                  Optimizer& optimizer) {
//     std::cout << "TEST " << test_name << ":\n";
//     auto start = std::chrono::system_clock::now();
//     net.fit(train_dataset, train_loss, n_of_epochs, batch_size, optimizer);
//     auto end = std::chrono::system_clock::now();
//     double loss = net.loss(test_dataset, test_loss);
//     auto time =
//         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
//             .count();
//     std::cout << "    time: " << time / 1000 << "." << time % 1000
//               << " s\n    train loss: " << net.loss(train_dataset,
//               train_loss)
//               << "\n    train accuracy: " << net.accuracy(train_dataset)
//               << "\n    test loss: " << loss
//               << "\n    test accuracy: " << net.accuracy(test_dataset)
//               << "\n    epochs: " << n_of_epochs
//               << "\n    batch size: " << batch_size
//               << "\n    optimizer: " << optimizer->describe() << "\n";
//     return loss;
// }

// void test_echo() {
// Net net1(Linear{GivensLayer(In(1), Out(1), Matrix{{0.5, 0.5}})},
//          NonLinear::Id());
// Net net2(Linear{MatrixLayer(In(1), Out(1), Matrix{{0.5, 0.5}})},
//          NonLinear::Id());
// Net net3(Linear{HouseholderLayer(In(1), Out(1), Matrix{{0.5, 0.5}})},
//          NonLinear::Id());
// std::cout << net1.predict(Matrix{{1}}) << net1.predict(Matrix{{2}})
//           << net1.predict(Matrix{{3}}) << "\n";
// std::cout << net2.predict(Matrix{{1}}) << net2.predict(Matrix{{2}})
//           << net2.predict(Matrix{{3}}) << "\n";
// std::cout << net3.predict(Matrix{{1}}) << " " <<
// net3.predict(Matrix{{2}})
//           << " " << net3.predict(Matrix{{3}}) << "\n";
// DataLoader data_loader(Matrix{{1, 2, 3, 4, 5, 6, 7, 8}},
//                        Matrix{{1, 2, 3, 4, 5, 6, 7, 8}});
// Net net(Linear{GivensLayer(In(1), Out(1), {0.9, 0.1})},
//         NonLinear::LeakyReLU());
// Optimizer optimizer = Constant(0.2);
// for (int q = 0; q < 10; ++q) {
//     net.fit(data_loader, LossFunction::Euclid(), 1, 10, optimizer);
//     for (int i = 0; i < 10; ++i) {
//         std::cout << net.predict(Matrix{{(double)i}})(0, 0) << "\n";
//     }
//     std::cout << "===\n";
// }
// }

// void test_sum() {
//     std::vector<TrainUnit> dataset{
//         {{1, 1}, {2}}, {{1, 2}, {3}}, {{1, 3}, {4}}, {{1, 4}, {5}},
//         {{2, 1}, {3}}, {{2, 2}, {4}}, {{2, 3}, {5}}, {{2, 4}, {6}},
//         {{3, 1}, {4}}, {{3, 2}, {5}}, {{3, 3}, {6}}, {{3, 4}, {7}},
//         {{4, 1}, {5}}, {{4, 2}, {6}}, {{4, 3}, {7}}, {{4, 4}, {8}}};
//     Random rnd;
//     Net net(Linear{GivensLayer(rnd.generateKaiming(2, 1), 2, 1)},
//             NonLinear::LeakyReLU());
//     Optimizer optimizer = Constant(net.linearLayers(), 0.07);
//     simple_test_loss("SUM", net, dataset, LossFunction::Euclid(), dataset,
//                      LossFunction::Euclid(), 100, 16, optimizer);
// }

// void test_sum_multi_layers() {
//     std::vector<TrainUnit> dataset{
//         {{1, 1}, {2}}, {{1, 2}, {3}}, {{1, 3}, {4}}, {{1, 4}, {5}},
//         {{2, 1}, {3}}, {{2, 2}, {4}}, {{2, 3}, {5}}, {{2, 4}, {6}},
//         {{3, 1}, {4}}, {{3, 2}, {5}}, {{3, 3}, {6}}, {{3, 4}, {7}},
//         {{4, 1}, {5}}, {{4, 2}, {6}}, {{4, 3}, {7}}, {{4, 4}, {8}}};
//     Random rnd;
//     Net net(Linear{GivensLayer(rnd.generateKaiming(2, 3), 2, 3)},
//             NonLinear::LeakyReLU());
//     net.AddLayer(Linear{GivensLayer(rnd.generateKaiming(3, 1), 3, 1)},
//                  NonLinear::LeakyReLU());
//     Optimizer optimizer = Constant(net.linearLayers(), 0.015);
//     simple_test_loss("SUM MULTI LAYERS", net, dataset,
//     LossFunction::Euclid(),
//                      dataset, LossFunction::Euclid(), 100, 1, optimizer);
// }

// void test_square() {
//     std::vector<TrainUnit> train = {{{0.1}, {0.01}},
//                                     {{0.2}, {0.04}},
//                                     {{0.3}, {0.09}},
//                                     {{0.4}, {0.16}},
//                                     {{0.5}, {0.25}}};
//     Random rnd;
//     Net net(Linear{GivensLayer(rnd.generateKaiming(1, 5), 1, 5)},
//             NonLinear::LeakyReLU());
//     net.AddLayer(Linear{GivensLayer(rnd.generateKaiming(5, 1), 5, 1)},
//     NonLinear::Id()); Optimizer optimizer = Constant(net.linearLayers(),
//     0.001); simple_test_loss("SQUARE", net, train, LossFunction::Euclid(),
//     train,
//                      LossFunction::Euclid(), 1000, 10, optimizer);
// }

// void test_mnist() {
//     std::vector<TrainUnit> train =
//         parseMNISTDataset("../train-images-idx3-ubyte/train-images.idx3-ubyte",
//                           "../train-labels-idx1-ubyte/train-labels.idx1-ubyte");
//     std::vector<TrainUnit> test =
//         parseMNISTDataset("../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
//                           "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
//     std::vector<int> seeds = {542,  2345, 5674, 5423, 64,
//                               2435, 765,  798,  5234, 23};
//     for (int seed : seeds) {
//         Random rnd(seed);
//         size_t input_size = 784;
//         size_t hid_size = 32;
//         size_t output_size = 10;
//         Vector w0 = rnd.generateKaiming(input_size, hid_size);
//         Net givens_net(Linear{GivensLayer(w0, input_size, hid_size)},
//                        NonLinear::LeakyReLU());
//         Net matrix_net(Linear{MatrixLayer(w0, input_size, hid_size)},
//                        NonLinear::LeakyReLU());
//         Net householder_net(Linear{HouseholderLayer(w0, input_size,
//         hid_size)},
//                             NonLinear::LeakyReLU());
//         Vector w1 = rnd.generateXavier(hid_size, output_size);
//         givens_net.AddLayer(Linear{GivensLayer(w1, hid_size, output_size)},
//                             NonLinear::Sigmoid());
//         matrix_net.AddLayer(Linear{MatrixLayer(w1, hid_size, output_size)},
//                             NonLinear::Sigmoid());
//         householder_net.AddLayer(
//             Linear{HouseholderLayer(w1, hid_size, output_size)},
//             NonLinear::Sigmoid());
//         double step = 0.01;
//         Optimizer givens_opt = Constant(givens_net.linearLayers(), step);
//         Optimizer matrix_opt = Constant(matrix_net.linearLayers(), step);
//         Optimizer householder_opt =
//             Constant(householder_net.linearLayers(), step);
//         double curr_loss = simple_test_loss_accuracy(
//             "MNIST GIVENS", givens_net, train, LossFunction::Euclid(), test,
//             LossFunction::Euclid(), 1, 10, givens_opt);
//         double curr_loss_2 = simple_test_loss_accuracy(
//             "MNIST MATRIX", matrix_net, train, LossFunction::Euclid(), test,
//             LossFunction::Euclid(), 1, 10, matrix_opt);
//         double curr_loss_3 = simple_test_loss_accuracy(
//             "MNIST HOUSEHOLDER", householder_net, train,
//             LossFunction::Euclid(), test, LossFunction::Euclid(), 1, 10,
//             householder_opt);
//     }
// }

void report_mnist() {
    DataLoader train =
        parseMNISTDataset("../train-images-idx3-ubyte/train-images.idx3-ubyte",
                          "../train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    DataLoader test =
        parseMNISTDataset("../t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
                          "../t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
    // std::vector<int> seeds = {542, 2345, 5674};
    std::vector<int> seeds = {542};
    LossFunction loss = LossFunction::Euclid();
    // std::vector<std::vector<int>> architectures = {
    //     {784, 32, 10}, {784, 10, 10}, {784, 2, 10}};
    std::vector<std::vector<int>> architectures = {{784, 32, 10}};
    std::vector<Optimizer> optimizers = {Optimizer{Constant(0.1)},
                                         Optimizer{Momentum(0.1, 0.5)},
                                         Optimizer{Adam(0.05)}};
    size_t batch_size = 6;
    double step = 0.1;
    size_t n_of_epochs = 5;
    for (const std::vector<int>& architecture : architectures) {
        for (const Optimizer& optimizer : optimizers) {
            std::cout << "=====EXPERIMENT=====\n";
            ClassificationReport report = getClassificationReportForGivensNets(
                architecture, seeds, train, test, loss, loss, batch_size,
                n_of_epochs, optimizer);
            printReport(report);
        }
    }
}

// void report_titanic() {
//     std::vector<TrainUnit> train;
//     std::vector<TrainUnit> test;
//     parseTitanicDataset("../titanic/Titanic-Dataset.csv", train, test);
//     std::vector<int> seeds = {42};
//     std::vector<LossFunction> losses = {LossFunction::Euclid()};
//     std::vector<Vector> architectures = {{9, 30, 2}, {9, 1000, 2}, {9, 10, 2}};
//     size_t batch_size = 8;
//     double step = 0.0001;
//     size_t n_of_epochs = 5;
//     for (Vector architecture : architectures) {
//         for (LossFunction loss : losses) {
//             for (size_t optim = 0; optim < 3; ++optim) {
//                 for (int seed : seeds) {
//                     std::cout << "=====EXPERIMENT=====\n";
//                     Random rnd(seed);
//                     size_t input_size = architecture[0];
//                     size_t hid_size = architecture[1];
//                     size_t output_size = architecture[2];
//                     Vector w0 = rnd.generateKaiming(input_size, hid_size);
//                     Net givens_net(
//                         Linear{GivensLayer(w0, input_size, hid_size)},
//                         NonLinear::LeakyReLU());
//                     Net matrix_net(
//                         Linear{MatrixLayer(w0, input_size, hid_size)},
//                         NonLinear::LeakyReLU());
//                     Net householder_net(
//                         Linear{HouseholderLayer(w0, input_size, hid_size)},
//                         NonLinear::LeakyReLU());
//                     Vector w1 = rnd.generateXavier(hid_size, output_size);
//                     givens_net.AddLayer(
//                         Linear{GivensLayer(w1, hid_size, output_size)},
//                         NonLinear::Sigmoid());
//                     matrix_net.AddLayer(
//                         Linear{MatrixLayer(w1, hid_size, output_size)},
//                         NonLinear::Sigmoid());
//                     householder_net.AddLayer(
//                         Linear{HouseholderLayer(w1, hid_size, output_size)},
//                         NonLinear::Sigmoid());
//                     Optimizer givens_opt;
//                     Optimizer matrix_opt;
//                     Optimizer householder_opt;
//                     if (optim == 0) {
//                         givens_opt = Constant(givens_net.linearLayers(), step);
//                         matrix_opt = Constant(matrix_net.linearLayers(), step);
//                         householder_opt =
//                             Constant(householder_net.linearLayers(), step);
//                     } else if (optim == 1) {
//                         givens_opt =
//                             Momentum(givens_net.linearLayers(), step, 0.5);
//                         matrix_opt =
//                             Momentum(matrix_net.linearLayers(), step, 0.5);
//                         householder_opt =
//                             Momentum(householder_net.linearLayers(), step, 0.5);
//                     } else {
//                         givens_opt = Adam(givens_net.linearLayers(), step);
//                         matrix_opt = Adam(matrix_net.linearLayers(), step);
//                         householder_opt =
//                             Adam(householder_net.linearLayers(), step);
//                     }
//                     CommonMetrics givens_metrics =
//                         measure(givens_net, train, loss, batch_size, givens_opt,
//                                 n_of_epochs);
//                     BinaryClassificationReport givens_report =
//                         getBinaryClassificationReport(givens_metrics,
//                                                       givens_net, train, loss,
//                                                       test, loss);
//                     printReport(givens_report);

//                     CommonMetrics matrix_metrics =
//                         measure(matrix_net, train, loss, batch_size, matrix_opt,
//                                 n_of_epochs);
//                     BinaryClassificationReport matrix_report =
//                         getBinaryClassificationReport(matrix_metrics,
//                                                       matrix_net, train, loss,
//                                                       test, loss);
//                     printReport(matrix_report);

//                     CommonMetrics householder_metrics =
//                         measure(householder_net, train, loss, batch_size,
//                                 householder_opt, n_of_epochs);
//                     BinaryClassificationReport householder_report =
//                         getBinaryClassificationReport(householder_metrics,
//                                                       householder_net, train,
//                                                       loss, test, loss);
//                     printReport(householder_report);
//                 }
//             }
//         }
//     }
// }

void report_boston() {
    Net::TrainTestLoaders loaders = parseBostonDataset("../boston/housing.csv");
    DataLoader train = std::move(loaders.train_dataloader);
    DataLoader test = std::move(loaders.test_dataloader);
    std::vector<int> seeds = {542, 2345, 5674};
    LossFunction loss = LossFunction::Euclid();
    std::vector<std::vector<int>> architectures = {
        {13, 100, 1}, {13, 1000, 1}, {13, 10, 1}};
    size_t batch_size = 1000;
    double step = 0.5;
    size_t n_of_epochs = 5;
    for (const std::vector<int>& architecture : architectures) {
        for (size_t optim = 0; optim < 1; ++optim) {
            for (int seed : seeds) {
                std::cout << "=====EXPERIMENT=====\n";
                Random rnd(seed);
                size_t input_size = architecture[0];
                size_t hid_size = architecture[1];
                size_t output_size = architecture[2];
                std::vector<double> w0 =
                    rnd.generateKaiming(In(input_size), Out(hid_size));
                Net givens_net(
                    Linear{GivensLayer(In(input_size), Out(hid_size), w0)},
                    NonLinear::LeakyReLU());
                Net matrix_net(
                    Linear{MatrixLayer(In(input_size), Out(hid_size), w0)},
                    NonLinear::LeakyReLU());
                Net householder_net(
                    Linear{HouseholderLayer(In(input_size), Out(hid_size), w0)},
                    NonLinear::LeakyReLU());
                std::vector<double> w1 =
                    rnd.generateXavier(In(hid_size), Out(output_size));
                givens_net.addLayer(
                    Linear{GivensLayer(In(hid_size), Out(output_size), w1)},
                    NonLinear::Sigmoid());
                matrix_net.addLayer(
                    Linear{MatrixLayer(In(hid_size), Out(output_size), w1)},
                    NonLinear::Sigmoid());
                householder_net.addLayer(
                    Linear{
                        HouseholderLayer(In(hid_size), Out(output_size), w1)},
                    NonLinear::Sigmoid());
                Optimizer givens_opt;
                Optimizer matrix_opt;
                Optimizer householder_opt;
                if (optim == 0) {
                    givens_opt = Constant(step);
                    matrix_opt = Constant(step);
                    householder_opt = Constant(step);
                } else if (optim == 1) {
                    // givens_opt = Momentum(step, 0.5);
                    // matrix_opt = Momentum(step, 0.5);
                    // householder_opt = Momentum(step, 0.5);
                } else {
                    // givens_opt = Adam(step);
                    // matrix_opt = Adam(step);
                    // householder_opt = Adam(step);
                }
                Vector grads = givens_net.fitAndGetMeanGradNorms(
                    train, loss, n_of_epochs, batch_size, givens_opt);
                // for (size_t epoch = 1; epoch <= n_of_epochs; ++epoch) {
                //     CommonMetrics givens_metrics = measure(
                //         givens_net, train, loss, batch_size, givens_opt,
                //         epoch);
                //     RegressionReport givens_report = getRegressionReport(
                //         givens_metrics, givens_net, train, loss, test, loss);
                //     printReport(givens_report);

                //     CommonMetrics matrix_metrics = measure(
                //         matrix_net, train, loss, batch_size, matrix_opt,
                //         epoch);
                //     RegressionReport matrix_report = getRegressionReport(
                //         matrix_metrics, matrix_net, train, loss, test, loss);
                //     printReport(matrix_report);

                //     CommonMetrics householder_metrics =
                //         measure(householder_net, train, loss, batch_size,
                //                 householder_opt, epoch);
                //     RegressionReport householder_report =
                //     getRegressionReport(
                //         householder_metrics, householder_net, train, loss,
                //         test, loss);
                //     printReport(householder_report);
                // }
            }
        }
    }
}

// void run_all_tests() {
//     test_echo();
//     test_sum();
//     test_sum_multi_layers();
//     test_square();
//     test_mnist();
// }

void run_all_reports() {
    report_mnist();
    // report_titanic();
    // report_boston();
    // test_echo();
}

}  // namespace neural_network
