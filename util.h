#pragma once

#include "CustomTypes.h"

namespace neural_network {
namespace util {
Eigen::JacobiSVD<Matrix> getSVD(In in, Out out, std::vector<double> weights);
void multFirstElemsOfColumnsByVectorElemwise(Matrix& a, const Vector& v);
void changeNumberOfRows(Matrix& a, Index new_number_of_rows);
}  // namespace util
}  // namespace neural_network
