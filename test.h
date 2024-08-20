#pragma once

#include "GivensNet.h"

namespace neural_network {

void simple_test_loss(const std::string& test_name, GivensNet& net,
                      const std::vector<TrainUnit>& dataset,
                      const LossFunction& loss, size_t n_of_epochs,
                      int batch_size, double step);
void test_echo();
void test_sum();
void test_sum_multi_layers();
void run_all_tests();

}  // namespace neural_network