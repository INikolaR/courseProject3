#include "Momentum.h"

#include <cassert>

#include "util.h"

namespace neural_network {
Momentum::Momentum(double step, double momentum_step)
    : step_(step), momentum_step_(momentum_step) {
}

Vector Momentum::fitAndGetMeanGradNorms(
    const DataLoader& data_loader, const LossFunction& loss, size_t n_of_epochs,
    size_t batch_size, std::vector<Linear>* linear_layers,
    std::vector<NonLinear>* non_linear_layers) const {
    assert(linear_layers->size() == non_linear_layers->size() &&
           "bad layer vectors");
    assert(n_of_epochs > 0 && "bad number of epochs");
    assert(batch_size > 0 && "bad batch size");

    Vector sum_grad_norms = Vector::Zero(linear_layers->size());
    std::vector<Array> h;
    for (size_t i = linear_layers->size(); i > 0; --i) {
        MatrixShape shape = (*linear_layers)[i - 1]->getGradShape();
        h.emplace_back(Array::Zero(shape.rows, shape.cols));
    }
    for (size_t i = 0; i < n_of_epochs; ++i) {
        sum_grad_norms += trainOneEpochAndGetMeanGradNorms(
            data_loader, loss, batch_size, linear_layers, non_linear_layers,
            &h);
    }
    return sum_grad_norms / n_of_epochs;
}

std::string Momentum::describe() const {
    std::stringstream ss;
    ss << "Momentum(step=" << step_ << ",m=" << momentum_step_ << ")";
    return ss.str();
}

Vector Momentum::trainOneEpochAndGetMeanGradNorms(
    const DataLoader& data_loader, const LossFunction& loss, size_t batch_size,
    std::vector<Linear>* linear_layers,
    std::vector<NonLinear>* non_linear_layers, std::vector<Array>* h) const {
    std::vector<TrainUnit> dataset = data_loader.getDataset(batch_size);
    Vector sum_grad_norms = Vector::Zero(linear_layers->size());
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
            util::changeNumberOfRows(result, (*linear_it)->sizeOut());
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
            util::changeNumberOfRows(u, (*linear_layer_it)->sizeOut());
            u = u.array() * non_linear_layer_it
                                ->evaluate1(non_linear_in_it->block(
                                    0, 0, u.rows(), u.cols()))
                                .array();
            Matrix g =
                (*linear_layer_it)
                    ->backwardCalcGradient(u, *linear_in_it, *non_linear_in_it);
            gradients.emplace_back(g);
        }
        for (size_t i = 0; i < gradients.size(); ++i) {
            sum_grad_norms(sum_grad_norms.rows() - i - 1) +=
                gradients[i].norm();
        }
        update(gradients, linear_layers, h);
    }
    return sum_grad_norms / dataset.size();
}

void Momentum::update(const std::vector<Matrix>& grads,
                      std::vector<Linear>* linear_layers,
                      std::vector<Array>* h) const {
    auto it_layers = linear_layers->begin();
    auto it_g = grads.rbegin();
    auto it_h = h->rbegin();
    for (; it_layers != linear_layers->end() && it_g != grads.rend();
         ++it_layers, ++it_g, ++it_h) {
        *it_h = (step_ * *it_g).array() + momentum_step_ * *it_h;
        (*it_layers)->update(*it_h, 1);
    }
}
}  // namespace neural_network
