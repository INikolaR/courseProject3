#pragma once

#include "GivensNet.h"

namespace neural_network {

void simple_test_loss(const std::string& test_name, GivensNet& net,
                      const std::vector<TrainUnit>& train_dataset,
                      const LossFunction& train_loss,
                      const std::vector<TrainUnit>& test_dataset,
                      const LossFunction& test_loss, size_t n_of_epochs,
                      int batch_size, double step);
void simple_test_loss_accuracy(const std::string& test_name, GivensNet& net,
                          const std::vector<TrainUnit>& train_dataset,
                          const LossFunction& train_loss,
                          const std::vector<TrainUnit>& test_dataset,
                          size_t n_of_epochs, int batch_size, double step);
void test_echo();
void test_sum();
void test_sum_multi_layers();
void test_mnist();
void test_vector_output();
void run_all_tests();
void test_givens_layer();

}  // namespace neural_network
