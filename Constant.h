#pragma once
#include <list>
#include <vector>

#include "CAnyLayer.h"

namespace neural_network {
class Constant {
public:
    Constant(std::list<Linear>& linear_layers, double step);
    void update(const std::vector<Vector>& grads);
    std::string describe() const;

private:
    std::list<Linear>& linear_layers_;
    double step_;
};
}  // namespace neural_network
