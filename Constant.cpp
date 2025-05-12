#include "Constant.h"

#include <iostream>

namespace neural_network {
Constant::Constant(double step) : step_(step) {
}

void Constant::fit(const DataLoader& data_loader, const LossFunction& loss,
                   size_t n_of_epochs, int batch_size,
                   std::vector<Linear>* linear_layers,
                   std::vector<NonLinear>* non_linear_layers) const {
    assert(linear_layers->size() == non_linear_layers->size() &&
           "bad layer vectors");
    for (size_t i = 0; i < n_of_epochs; ++i) {
        trainOneEpoch(data_loader, loss, batch_size, linear_layers,
                      non_linear_layers);
    }
}

std::string Constant::describe() const {
    std::stringstream ss;
    ss << "Constant(step=" << step_ << ")";
    return ss.str();
}
void Constant::trainOneEpoch(const DataLoader& data_loader,
                             const LossFunction& loss, int batch_size,
                             std::vector<Linear>* linear_layers,
                             std::vector<NonLinear>* non_linear_layers) const {
    std::vector<TrainUnit> dataset = data_loader.getDataset(batch_size);
    for (size_t i = 0; i < dataset.size(); ++i) {
        TrainUnit batch = dataset[i];
        std::vector<Matrix> linear_in;
        std::vector<Matrix> non_linear_in;
        Matrix result = batch.x;
        auto linear_it = linear_layers->begin();
        auto non_linear_it = non_linear_layers->begin();
        for (; linear_it != linear_layers->end();
             ++linear_it, ++non_linear_it) {
            linear_in.emplace_back(result);
            result = (*linear_it)->forwardOnTrain(result);
            non_linear_in.emplace_back(result);
            changeNumberOfRows(result, (*linear_it)->sizeOut());
            result = non_linear_it->evaluate0(result);
        }
        Matrix u = loss.evaluate1(result, batch.y);

        std::vector<Matrix> gradients;

        auto linear_layer_it = linear_layers->rbegin();
        auto non_linear_layer_it = non_linear_layers->rbegin();
        auto non_linear_in_it = non_linear_in.rbegin();
        auto linear_in_it = linear_in.rbegin();
        for (; linear_layer_it != linear_layers->rend();
             ++linear_layer_it, ++non_linear_layer_it, ++non_linear_in_it,
             ++linear_in_it) {
            changeNumberOfRows(u, (*linear_layer_it)->sizeOut());
            u = u.array() * non_linear_layer_it
                                ->evaluate1(non_linear_in_it->block(
                                    0, 0, u.rows(), u.cols()))
                                .array();
            Matrix g =
                (*linear_layer_it)
                    ->backwardCalcGradient(u, *linear_in_it, *non_linear_in_it);
            gradients.emplace_back(g);
        }

        update(gradients, linear_layers);
    }
}

void Constant::update(const std::vector<Matrix>& grads,
                      std::vector<Linear>* linear_layers) const {
    auto it_layers = linear_layers->begin();
    auto it_g = grads.rbegin();
    for (; it_layers != linear_layers->end() && it_g != grads.rend();
         ++it_layers, ++it_g) {
        (*it_layers)->update(*it_g, step_);
    }
}
}  // namespace neural_network
