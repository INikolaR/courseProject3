#include "test.h"

#include <iostream>

namespace neural_network {

void test_echo() {
    std::vector<TrainUnit> dataset{{{1}, {1}}, {{2}, {2}}, {{3}, {3}},
                                   {{4}, {4}}, {{5}, {5}}, {{6}, {6}},
                                   {{7}, {7}}, {{8}, {8}}};
    GivensNet net(1, 1, ActivationFunction::LeakyReLU());
    net.fit(dataset, LossFunction::Euclid(), 100, 1, 0.02);
    std::cout << "10 -> " << net.predict({10})[0] << std::endl;
}

void test_sum() {
    std::vector<TrainUnit> dataset{
        {{1, 1}, {2}}, {{1, 2}, {3}}, {{1, 3}, {4}}, {{1, 4}, {5}},
        {{2, 1}, {3}}, {{2, 2}, {4}}, {{2, 3}, {5}}, {{2, 4}, {6}},
        {{3, 1}, {4}}, {{3, 2}, {5}}, {{3, 3}, {6}}, {{3, 4}, {7}},
        {{4, 1}, {5}}, {{4, 2}, {6}}, {{4, 3}, {7}}, {{4, 4}, {8}}};
    GivensNet net(2, 1, ActivationFunction::LeakyReLU());
    net.fit(dataset, LossFunction::Euclid(), 100, 1, 0.05);
    std::cout << "1.7 + 2.7 = " << net.predict({1.7, 2.7})[0] << std::endl;
}

void run_all_tests() {
    // test_echo();
    test_sum();
}

}  // namespace neural_network
