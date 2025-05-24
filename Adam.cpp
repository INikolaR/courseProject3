#include <cassert>

#include "Adam.h"

namespace neural_network {
Adam::Adam(double step) : Adam(step, 0.9, 0.999, 1e-8) {
}

Adam::Adam(double step, double beta1, double beta2, double epsilon)
    : step_(step), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
}

Vector Adam::fitAndGetMeanGradNorms(
    const DataLoader& data_loader, const LossFunction& loss, size_t n_of_epochs,
    size_t batch_size, std::vector<Linear>* linear_layers,
    std::vector<NonLinear>* non_linear_layers) const {
    assert(linear_layers->size() == non_linear_layers->size() &&
           "bad layer vectors");
    assert(n_of_epochs > 0 && "bad number of epochs");
    assert(batch_size > 0 && "bad batch size");

    Vector sum_grad_norms = Vector::Zero(linear_layers->size());
    std::vector<Array> m;
    std::vector<Array> v;
    for (size_t i = linear_layers->size(); i > 0; --i) {
        m.emplace_back(Array::Zero((*linear_layers)[i - 1]->size()));
        v.emplace_back(Array::Zero((*linear_layers)[i - 1]->size()));
    }
    double beta1_cum = 1;
    double beta2_cum = 1;
    for (size_t i = 0; i < n_of_epochs; ++i) {
        sum_grad_norms += trainOneEpochAndGetMeanGradNorms(
            data_loader, loss, batch_size, linear_layers, non_linear_layers, &m,
            &v, &beta1_cum, &beta2_cum);
    }
    return sum_grad_norms / n_of_epochs;
}

std::string Adam::describe() const {
    std::stringstream ss;
    ss << "Adam(step=" << step_ << ",b1=" << beta1_ << ",b2=" << beta2_
       << ",e=" << epsilon_ << ")";
    return ss.str();
}

Vector Adam::trainOneEpochAndGetMeanGradNorms(
    const DataLoader& data_loader, const LossFunction& loss, size_t batch_size,
    std::vector<Linear>* linear_layers,
    std::vector<NonLinear>* non_linear_layers, std::vector<Array>* m,
    std::vector<Array>* v, double* beta1_cum, double* beta2_cum) const {
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
        for (size_t i = 0; i < gradients.size(); ++i) {
            sum_grad_norms(sum_grad_norms.rows() - i - 1) +=
                gradients[i].norm();
        }
        update(gradients, linear_layers, m, v, beta1_cum, beta2_cum);
    }
    return sum_grad_norms / dataset.size();
}

void Adam::update(const std::vector<Matrix>& grads,
                  std::vector<Linear>* linear_layers, std::vector<Array>* m,
                  std::vector<Array>* v, double* beta1_cum,
                  double* beta2_cum) const {
    auto it_layers = linear_layers->begin();
    auto it_g = grads.rbegin();
    auto it_m = m->rbegin();
    auto it_v = m->rbegin();
    for (; it_layers != linear_layers->end() && it_g != grads.rend();
         ++it_layers, ++it_g, ++it_m, ++it_v) {
        Array g_array = it_g->array();
        *it_m = beta1_ * *it_m + (1 - beta1_) * g_array;
        *it_v = beta2_ * *it_v + (1 - beta2_) * (g_array.square());
        *beta1_cum *= beta1_;
        *beta2_cum *= beta2_;
        (*it_layers)
            ->update(*it_m / (it_v->sqrt() + epsilon_) /
                         ((1 - *beta1_cum) / (1 - *beta2_cum)),
                     step_);
    }
}
}  // namespace neural_network
