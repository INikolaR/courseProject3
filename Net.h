#pragma once
#include <list>

#include "DataLoader.h"
#include "Linear.h"
#include "LossFunction.h"
#include "NonLinear.h"
#include "Optimizer.h"

namespace neural_network {

class Net {
public:
    Net(Linear l, NonLinear f);
    void AddLayer(Linear l, NonLinear f);
    Matrix predict(const Matrix& x) const;
    void fit(const DataLoader& data_loader, const LossFunction& loss,
             size_t n_of_epochs, int batch_size, const Optimizer& optimizer);
    // double loss(const std::vector<TrainUnit>& test_dataset,
    //             const LossFunction& loss) const;
    // double accuracy(const std::vector<TrainUnit> test_dataset) const;
    // Vector trainOneEpochWithFrobeniusNorms(
    //     const std::vector<TrainUnit>& dataset, const LossFunction& loss,
    //     int batch_size, Optimizer& optimizer);
    std::string describe() const;

private:
    // void trainOneEpoch(const std::vector<TrainUnit>& dataset,
    //                    const LossFunction& loss, int batch_size,
    //                    Optimizer& optimizer);
    // void trainOneBatch(const std::vector<TrainUnit>::const_iterator begin,
    //                    const std::vector<TrainUnit>::const_iterator end,
    //                    const LossFunction& loss, Optimizer& optimizer);
    // void trainOneBatchWithAddingFrobeniusNorms(
    //     const std::vector<TrainUnit>::const_iterator begin,
    //     const std::vector<TrainUnit>::const_iterator end,
    //     const LossFunction& loss, Optimizer& optimizer,
    //     Vector& frobenius_norms);
    // std::vector<Vector> trainOneUnit(const Vector& x, const Vector& y,
    //                                  const LossFunction& loss);
    // void addGradients(std::vector<Vector>& a, const std::vector<Vector>& b);
    Index in_;
    Index out_;
    std::vector<Linear> linear_layers_;
    std::vector<NonLinear> non_linear_layers_;
};

}  // namespace neural_network
