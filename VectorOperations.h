#pragma once

#include "CustomTypes.h"
#include "eigen/Eigen/Dense"

namespace neural_network {
void GivensRotation(double angle, Index row, Matrix& v);
void GivensRotation(double sin, double cos, Index row, Matrix& v);
void HouseholderReflection(const Vector& u, Matrix& a);
void HouseholderReflection(const Vector& u, Matrix& a, Index a_rows);
SVD getGivensPerfomance(In in, Out out, const std::vector<double>& m);
SVD getHouseholderPerfomance(In in, Out out, const std::vector<double>& m);

void multFirstElemsOfColumnsByVectorElemwise(Matrix& a, const Vector& v);
void changeNumberOfRows(Matrix& a, Index new_number_of_rows);

}  // namespace neural_network
