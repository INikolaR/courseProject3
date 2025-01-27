#pragma once

#include <list>
#include <string>
#include <vector>

#include "CAnyLayer.h"
#include "CustomTypes.h"

namespace neural_network {
class Adam {
public:
    Adam(std::list<Linear>& linear_layers, double step);
    Adam(std::list<Linear>& linear_layers, double step, double beta1, double beta2, double epsilon);
    void update(const std::vector<Vector>& grads);
    std::string describe() const;

private:
    std::vector<Vector> zerosInversed(std::list<Linear>& parameters);

    std::list<Linear>& linear_layers_;
    double step_;
    double beta1_;
    double beta2_;
    double beta1_k_;
    double beta2_k_;
    double epsilon_;
    std::vector<Vector> m_;
    std::vector<Vector> v_;
};
}  // namespace neural_network
