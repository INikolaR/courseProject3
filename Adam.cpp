#include "Adam.h"

namespace neural_network {
Adam::Adam(std::list<Linear>& linear_layers, double step)
    : Adam(linear_layers, step, 0.9, 0.999, 1e-8) {
}

Adam::Adam(std::list<Linear>& linear_layers, double step, double beta1,
           double beta2, double epsilon)
    : linear_layers_(linear_layers),
      step_(step),
      beta1_(beta1),
      beta2_(beta2),
      beta1_k_(beta1),
      beta2_k_(beta2),
      epsilon_(epsilon),
      m_(std::move(zerosInversed(linear_layers))),
      v_(std::move(zerosInversed(linear_layers))) {
}

void Adam::update(const std::vector<Vector>& grads) {
    auto it_layers = linear_layers_.rbegin();
    auto it_g = grads.begin();
    auto it_m = m_.begin();
    auto it_v = v_.begin();
    Vector actual_grad;
    for (; it_layers != linear_layers_.rend();
         ++it_layers, ++it_g, ++it_m, ++it_v) {
        actual_grad.clear();
        actual_grad.reserve((*it_layers)->size());
        for (size_t i = 0; i < (*it_layers)->size(); ++i) {
            (*it_m)[i] = beta1_ * (*it_m)[i] + (1 - beta1_) * (*it_g)[i];
            (*it_v)[i] =
                beta2_ * (*it_v)[i] + (1 - beta2_) * (*it_g)[i] * (*it_g)[i];
            actual_grad.emplace_back(
                (*it_m)[i] / (1 - beta1_k_) /
                (sqrt((*it_v)[i] / (1 - beta2_k_)) + epsilon_));
        }
        (*it_layers)->update(actual_grad, step_);
    }
    beta1_k_ *= beta1_;
    beta2_k_ *= beta2_;
}

std::vector<Vector> Adam::zerosInversed(std::list<Linear>& linear_layers) {
    std::vector<Vector> zeros;
    zeros.reserve(linear_layers.size());
    for (auto it = linear_layers.rbegin(); it != linear_layers.rend(); ++it) {
        zeros.emplace_back(Vector((*it)->size(), 0));
    }
    return zeros;
}

std::string Adam::describe() const {
    std::stringstream ss;
    ss << "Adam(step=" << step_ << ",b1=" << beta1_ << ",b2=" << beta2_
       << ",e=" << epsilon_ << ")";
    return ss.str();
}
}  // namespace neural_network
