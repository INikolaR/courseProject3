#include "Constant.h"

namespace neural_network {
Constant::Constant(std::list<Linear>& linear_layers, double step)
    : linear_layers_(linear_layers), step_(step) {
}

void Constant::update(const std::vector<Vector>& grads) {
    auto it_layers = linear_layers_.begin();
    auto it_g = grads.rbegin();
    for (; it_layers != linear_layers_.end() && it_g != grads.rend();
         ++it_layers, ++it_g) {
        (*it_layers)->update(*it_g, step_);
    }
}

std::string Constant::describe() const {
    std::stringstream ss;
    ss << "Constant(step=" << step_ << ")";
    return ss.str();
}
}  // namespace neural_network
