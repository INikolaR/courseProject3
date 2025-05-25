#pragma once

#include "CustomTypes.h"
#include "EigenProxyTypes.h"

namespace neural_network {
class DataLoader {
public:
    DataLoader(Matrix x, Matrix y);

    std::vector<TrainUnit> getDataset(size_t batch_size) const;
    Index sizeIn() const;
    Index sizeOut() const;

private:
    Matrix x_;
    Matrix y_;
};

struct TrainTestLoaders {
    DataLoader train_dataloader;
    DataLoader test_dataloader;
};
}  // namespace neural_network
