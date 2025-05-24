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
    struct TrainTestLoaders {
        DataLoader train_dataloader;
        DataLoader test_dataloader;
    };

    Net(Linear l, NonLinear f);
    void addLayer(Linear l, NonLinear f);
    Matrix predict(const Matrix& x) const;
    Vector fitAndGetMeanGradNorms(const DataLoader& data_loader,
                                  const LossFunction& loss, size_t n_of_epochs,
                                  size_t batch_size,
                                  const Optimizer& optimizer);
    size_t getNumOfLayers() const;
    std::string describe() const;

private:
    Index in_;
    Index out_;
    std::vector<Linear> linear_layers_;
    std::vector<NonLinear> non_linear_layers_;
};

}  // namespace neural_network
