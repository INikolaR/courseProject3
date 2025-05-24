#pragma once

#include <vector>

#include "AnyObject.h"
#include "CustomTypes.h"
#include "DataLoader.h"

namespace NSDetail {
template <class TBase>
class IAnyOptimizer : public TBase {
public:
    virtual Eigen::VectorXd fitAndGetMeanGradNorms(
        const neural_network::DataLoader& data_loader,
        const neural_network::LossFunction& loss, size_t n_of_epochs,
        size_t batch_size, std::vector<Linear>* linear_layers,
        std::vector<neural_network::NonLinear>* non_linear_layers) const = 0;
    virtual std::string describe() const = 0;
};

template <class TBase, class TObject>
class CAnyOptimizerImpl : public TBase {
    // This using is for convenience only
    using CBase = TBase;

public:
    // We need to open all constructors of the base class
    using CBase::CBase;
    Eigen::VectorXd fitAndGetMeanGradNorms(
        const neural_network::DataLoader& data_loader,
        const neural_network::LossFunction& loss, size_t n_of_epochs,
        size_t batch_size, std::vector<Linear>* linear_layers,
        std::vector<neural_network::NonLinear>* non_linear_layers) const {
        return CBase::Object().fitAndGetMeanGradNorms(
            data_loader, loss, n_of_epochs, batch_size, linear_layers,
            non_linear_layers);
    }

    std::string describe() const {
        return CBase::Object().describe();
    }
};

// A using for convenience
using CAnyOptimizerT = CAnyObject<IAnyOptimizer, CAnyOptimizerImpl>;
}  // namespace NSDetail

class Optimizer : public NSDetail::CAnyOptimizerT {
    // This using is for convenience only
    using CBase = NSDetail::CAnyOptimizerT;

public:
    // We only need to open all constructors of the base class
    using CBase::CBase;
};
