#include "Momentum.h"

#include "VectorOperations.h"

namespace neural_network {
Momentum::Momentum(std::list<Linear>& linear_layers, double step,
                   double momentum_step)
    : linear_layers_(linear_layers),
      step_(step),
      m_(momentum_step),
      h_(std::move(zerosInversed(linear_layers_))) {
}

void Momentum::update(const std::vector<Vector>& grads) {
    auto it_layers = linear_layers_.rbegin();
    auto it_g = grads.begin();
    auto it_h = h_.begin();
    Vector actual_grad;
    for (; it_layers != linear_layers_.rend() && it_g != grads.end();
         ++it_layers, ++it_g, ++it_h) {
        actual_grad.clear();
        actual_grad.reserve(it_g->size());
        for (size_t i = 0; i < it_g->size(); ++i) {
            actual_grad.emplace_back(m_ * (*it_h)[i] + (1 - m_) * (*it_g)[i]);
        }
        (*it_layers)->update(actual_grad, step_);
    }
}

std::string Momentum::describe() const {
    std::stringstream ss;
    ss << "Momentum(step=" << step_ << ",m=" << m_ << ")";
    return ss.str();
}

std::vector<Vector> Momentum::zerosInversed(std::list<Linear>& linear_layers) {
    std::vector<Vector> zeros;
    zeros.reserve(linear_layers.size());
    for (auto it = linear_layers.rbegin(); it != linear_layers.rend(); ++it) {
        zeros.emplace_back(Vector((*it)->size(), 0));
    }
    return zeros;
}
}  // namespace neural_network
