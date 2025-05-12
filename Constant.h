#pragma once
#include <list>
#include <vector>

#include "DataLoader.h"
#include "Linear.h"
#include "LossFunction.h"
#include "NonLinear.h"

namespace neural_network {
class Constant {
public:
    Constant(double step);

    void fit(const DataLoader& data_loader, const LossFunction& loss,
             size_t n_of_epochs, int batch_size,
             std::vector<Linear>* linear_layers,
             std::vector<NonLinear>* non_linear_layers) const;

    std::string describe() const;

private:
    void trainOneEpoch(const DataLoader& data_loader, const LossFunction& loss,
                       int batch_size, std::vector<Linear>* linear_layers,
                       std::vector<NonLinear>* non_linear_layers) const;
    void update(const std::vector<Matrix>& grads,
                std::vector<Linear>* linear_layers) const;

    double step_;
};
}  // namespace neural_network
