#pragma once

#include "EigenProxyTypes.h"

namespace neural_network {
struct TrainUnit {
    Matrix x;
    Matrix y;
};
struct SVD {
    Vector U;
    Vector sigma;
    Vector V;
};
struct MatrixShape {
    Index rows;
    Index cols;
};
enum In : Index;
enum Out : Index;
}  // namespace neural_network
