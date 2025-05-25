#include "Net.h"

#include <cassert>

namespace neural_network {

Net::Net(Linear l, NonLinear f) : in_(l->sizeIn()), out_(l->sizeOut()) {
    linear_layers_.emplace_back(std::move(l));
    non_linear_layers_.emplace_back(std::move(f));
}

void Net::addLayer(Linear l, NonLinear f) {
    assert(l->sizeIn() == out_ && "bad input size of layer");
    out_ = l->sizeOut();
    linear_layers_.emplace_back(std::move(l));
    non_linear_layers_.emplace_back(std::move(f));
}

Matrix Net::predict(const Matrix& x) const {
    assert(x.rows() == in_ && "bad size of x");
    auto linear_it = linear_layers_.begin();
    auto non_linear_it = non_linear_layers_.begin();
    Matrix temp = (*linear_it)->forward(x);
    temp = non_linear_it->evaluate0(temp);
    ++linear_it;
    ++non_linear_it;
    for (; linear_it != linear_layers_.end(); ++linear_it, ++non_linear_it) {
        temp = (*linear_it)->forward(temp);
        temp = non_linear_it->evaluate0(temp);
    }
    return temp;
}

Vector Net::fitAndGetMeanGradNorms(const DataLoader& data_loader,
                                   const LossFunction& loss, size_t n_of_epochs,
                                   size_t batch_size,
                                   const Optimizer& optimizer) {
    assert(n_of_epochs > 0 && "bad number of epochs");
    assert(batch_size > 0 && "bad batch size");
    return optimizer->fitAndGetMeanGradNorms(data_loader, loss, n_of_epochs,
                                             batch_size, &linear_layers_,
                                             &non_linear_layers_);
}

size_t Net::getNumOfLayers() const {
    return linear_layers_.size();
}

std::string Net::describe() const {
    std::stringstream ss;
    auto linear_it = linear_layers_.begin();
    auto non_linear_it = non_linear_layers_.begin();
    ss << (*linear_it)->describe() << " -> " << non_linear_it->describe();
    ++linear_it;
    ++non_linear_it;
    for (; linear_it != linear_layers_.end(); ++linear_it, ++non_linear_it) {
        ss << " -> " << (*linear_it)->describe() << " -> "
           << non_linear_it->describe();
    }
    return ss.str();
}

}  // namespace neural_network
