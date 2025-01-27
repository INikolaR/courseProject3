#pragma once
#include <list>

#include "ActivationFunction.h"
#include "CAnyLayer.h"
#include "CAnyOptimizer.h"
#include "LossFunction.h"

namespace neural_network {

class Net {
public:
    Net(Linear l, ActivationFunction f);
    void AddLayer(Linear l, ActivationFunction f);
    Vector predict(const Vector& x) const;
    void fit(const std::vector<TrainUnit>& dataset, const LossFunction& loss,
             size_t n_of_epochs, int batch_size, Optimizer& optimizer);
    double loss(const std::vector<TrainUnit>& test_dataset,
                const LossFunction& loss) const;
    double accuracy(const std::vector<TrainUnit> test_dataset) const;
    Vector trainOneEpochWithFrobeniusNorms(
        const std::vector<TrainUnit>& dataset, const LossFunction& loss,
        int batch_size, Optimizer& optimizer);
    std::string describe() const;
    std::list<Linear>& linearLayers();

private:
    void trainOneEpoch(const std::vector<TrainUnit>& dataset,
                       const LossFunction& loss, int batch_size,
                       Optimizer& optimizer);
    void trainOneBatch(const std::vector<TrainUnit>::const_iterator begin,
                       const std::vector<TrainUnit>::const_iterator end,
                       const LossFunction& loss, Optimizer& optimizer);
    void trainOneBatchWithAddingFrobeniusNorms(
        const std::vector<TrainUnit>::const_iterator begin,
        const std::vector<TrainUnit>::const_iterator end,
        const LossFunction& loss, Optimizer& optimizer,
        Vector& frobenius_norms);
    std::vector<Vector> trainOneUnit(const Vector& x, const Vector& y,
                                     const LossFunction& loss);
    void addGradients(std::vector<Vector>& a, const std::vector<Vector>& b);
    size_t in_;
    size_t out_;
    std::list<Linear> linear_layers_;
    std::list<ActivationFunction> non_linear_layers_;
};

}  // namespace neural_network
