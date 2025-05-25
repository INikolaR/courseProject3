#pragma once

#include "DataLoader.h"
#include "Linear.h"
#include "LossFunction.h"
#include "NonLinear.h"

namespace neural_network {
class Constant {
public:
    Constant(double step);

    Vector fitAndGetMeanGradNorms(
        const DataLoader& data_loader, const LossFunction& loss,
        size_t n_of_epochs, size_t batch_size,
        std::vector<Linear>* linear_layers,
        std::vector<NonLinear>* non_linear_layers) const;
    std::string describe() const;

private:
    Vector trainOneEpochAndGetMeanGradNorms(
        const DataLoader& data_loader, const LossFunction& loss,
        size_t batch_size, std::vector<Linear>* linear_layers,
        std::vector<NonLinear>* non_linear_layers) const;
    void update(const std::vector<Matrix>& grads,
                std::vector<Linear>* linear_layers) const;

    double step_;
};
}  // namespace neural_network
