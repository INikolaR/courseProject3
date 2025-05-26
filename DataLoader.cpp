#include "Dataloader.h"

#include <cassert>

namespace neural_network {
DataLoader::DataLoader(Matrix x, Matrix y)
    : x_(std::move(x)), y_(std::move(y)) {
    assert(x_.cols() == y_.cols() && "different size of input and output");
}

std::vector<TrainUnit> DataLoader::getDataset(size_t batch_size) const {
    assert(batch_size > 0 && "bad batch size for dataset");
    std::vector<TrainUnit> dataset;
    size_t i = 0;
    for (; i + batch_size < x_.cols(); i += batch_size) {
        Matrix x_batch = x_.block(0, i, x_.rows(), batch_size).eval();
        Matrix y_batch = y_.block(0, i, y_.rows(), batch_size).eval();
        dataset.emplace_back(TrainUnit{x_batch, y_batch});
    }
    Matrix x_batch = x_.block(0, i, x_.rows(), x_.cols() - i).eval();
    Matrix y_batch = y_.block(0, i, y_.rows(), y_.cols() - i).eval();
    dataset.emplace_back(TrainUnit{x_batch, y_batch});
    return dataset;
}

Index DataLoader::sizeIn() const {
    return x_.rows();
}

Index DataLoader::sizeOut() const {
    return y_.rows();
}
}  // namespace neural_network
