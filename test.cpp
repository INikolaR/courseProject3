#include "test.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>

namespace neural_network {

void simple_test_loss(const std::string& test_name, GivensNet& net,
                      const std::vector<TrainUnit>& dataset,
                      const LossFunction& loss, size_t n_of_epochs,
                      int batch_size, double step) {
    std::cout << "TEST " << test_name << ":\n";
    auto start = std::chrono::system_clock::now();
    net.fit(dataset, loss, n_of_epochs, batch_size, step);
    auto end = std::chrono::system_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "    time: " << time / 1000 << "." << time % 1000
              << " s\n    loss: " << net.loss(dataset, loss)
              << "\n    epochs: " << n_of_epochs
              << "\n    batch size: " << batch_size << "\n    step: " << step
              << "\n";
}

void test_echo() {
    std::vector<TrainUnit> dataset{{{1}, {1}}, {{2}, {2}}, {{3}, {3}},
                                   {{4}, {4}}, {{5}, {5}}, {{6}, {6}},
                                   {{7}, {7}}, {{8}, {8}}};
    GivensNet net(1, 1, ActivationFunction::LeakyReLU());
    simple_test_loss("ECHO", net, dataset, LossFunction::Euclid(), 100, 1,
                     0.02);
}

void test_sum() {
    std::vector<TrainUnit> dataset{
        {{1, 1}, {2}}, {{1, 2}, {3}}, {{1, 3}, {4}}, {{1, 4}, {5}},
        {{2, 1}, {3}}, {{2, 2}, {4}}, {{2, 3}, {5}}, {{2, 4}, {6}},
        {{3, 1}, {4}}, {{3, 2}, {5}}, {{3, 3}, {6}}, {{3, 4}, {7}},
        {{4, 1}, {5}}, {{4, 2}, {6}}, {{4, 3}, {7}}, {{4, 4}, {8}}};
    GivensNet net(2, 1, ActivationFunction::LeakyReLU());
    simple_test_loss("SUM", net, dataset, LossFunction::Euclid(), 1000, 16,
                     0.05);
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
                     1000, 1, 0.015);
}

void run_all_tests() {
    test_echo();
    test_sum();
    test_sum_multi_layers();
}

}  // namespace neural_network
