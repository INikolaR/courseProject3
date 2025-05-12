#pragma once
#include "CustomTypes.h"

namespace neural_network {
class DataLoader {
public:
    DataLoader(Matrix x, Matrix y);

    std::vector<TrainUnit> getDataset(size_t batch_size) const;

private:
    Matrix x_;
    Matrix y_;
};
}  // namespace neural_network
