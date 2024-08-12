#include "test.h"

#include <iostream>

namespace neural_network {

void test_echo() {
    std::vector<TrainUnit> dataset{{{1}, {1}}, {{2}, {2}}, {{3}, {3}},
                                   {{4}, {4}}, {{5}, {5}}, {{6}, {6}},
                                   {{7}, {7}}, {{8}, {8}}};
    GivensNet net(1, 1, ActivationFunction::LeakyReLU());
    net.fit(dataset, LossFunction::Euclid(), 100, 0.02);
    std::cout << "10 -> " << net.predict({10})[0] << std::endl;
}

}  // namespace neural_network
