#pragma once

#include "Net.h"

namespace neural_network {

// void simple_test_loss(const std::string& test_name, Net& net,
//                       const std::vector<TrainUnit>& train_dataset,
//                       const LossFunction& train_loss,
//                       const std::vector<TrainUnit>& test_dataset,
//                       const LossFunction& test_loss, size_t n_of_epochs,
//                       int batch_size, Optimizer& optimizer);
// void simple_test_loss_accuracy(const std::string& test_name, Net& net,
//                                const std::vector<TrainUnit>& train_dataset,
//                                const LossFunction& train_loss,
//                                const std::vector<TrainUnit>& test_dataset,
//                                LossFunction& test_loss, size_t n_of_epochs,
//                                int batch_size, Optimizer& optimizer);
// void test_echo();
// void test_sum();
// void test_sum_multi_layers();
// void test_square();
// void test_mnist();
void report_mnist();
// void report_titanic();
void report_boston();
// void run_all_tests();
void run_all_reports();

}  // namespace neural_network
