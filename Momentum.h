#pragma once

#include <list>
#include <vector>

#include "CAnyLayer.h"

namespace neural_network {
class Momentum {
public:
    Momentum(std::list<Linear>& linear_layers, double step, double momentum_step);
    void update(const std::vector<Vector>& grads);
    std::string describe() const;

private:
    std::vector<Vector> zerosInversed(std::list<Linear>& parameters);

    std::list<Linear>& linear_layers_;
    double step_;
    double m_;
    std::vector<Vector> h_;
};
}  // namespace neural_network
