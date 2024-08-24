#pragma once
#include <list>

#include "GivensLayer.h"
#include "LossFunction.h"

namespace neural_network {

class GivensNet {
public:
    explicit GivensNet(size_t in, size_t out, ActivationFunction f);
    void AddLayer(size_t new_out, ActivationFunction f);
    Vector predict(const Vector& x) const;
    void fit(const std::vector<TrainUnit>& dataset, const LossFunction& loss,
             size_t n_of_epochs, int batch_size, double step);
    double loss(const std::vector<TrainUnit>& test_dataset,
                const LossFunction& loss) const;
    double accuracy(const std::vector<TrainUnit> test_dataset) const;

private:
    void trainOneEpoch(const std::vector<TrainUnit>& dataset,
                       const LossFunction& loss, int batch_size, double step);
    void trainOneBatch(const std::vector<TrainUnit>::const_iterator begin,
                       const std::vector<TrainUnit>::const_iterator end,
                       const LossFunction& loss, double step);
    std::vector<Gradient> trainOneUnit(const Vector& x, const Vector& y,
                                       const LossFunction& loss, double step);
    void addGradients(std::vector<Gradient>& a, const std::vector<Gradient>& b);
    size_t in_;
    size_t out_;
    std::list<GivensLayer> linear_layers_;
    std::list<ActivationFunction> non_linear_layers_;
};

}  // namespace neural_network
