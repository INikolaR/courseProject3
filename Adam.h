#pragma once

#include <list>
#include <string>
#include <vector>

#include "CustomTypes.h"
#include "DataLoader.h"
#include "Linear.h"
#include "LossFunction.h"
#include "NonLinear.h"

namespace neural_network {
class Adam {
public:
    using Array = Eigen::ArrayXd;

    Adam(double step);
    Adam(double step, double beta1, double beta2, double epsilon);

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
        std::vector<NonLinear>* non_linear_layers, std::vector<Array>* m,
        std::vector<Array>* v, double* beta1_cumulative,
        double* beta2_cumulative) const;
    void update(const std::vector<Matrix>& grads,
                std::vector<Linear>* linear_layers, std::vector<Array>* m,
                std::vector<Array>* v, double* beta1_cumulative,
                double* beta2_cumulative) const;

    double step_;
    double beta1_;
    double beta2_;
    double epsilon_;
};
}  // namespace neural_network
