#pragma once

#include "CustomTypes.h"
#include "eigen/Eigen/Dense"

namespace neural_network {

void operator+=(Vector& a, const Vector& b);
void operator-=(Vector& a, const Vector& b);
Vector operator-(const Vector& a, const Vector& b);
void operator*=(Vector& a, const Vector& b);
void operator*=(Vector& a, double b);
Vector operator*(const Vector& a, const Vector& b);
Vector operator*(double l, const Vector& b);
Vector operator*(const Vector& b, double l);
void updateVector(Vector& v, const Vector& dv, double step);
void updateReversedVector(Vector& v, const Vector& dv, double step);
double dot(const Vector& a, const Vector& b);
double dotn(Vector::const_iterator a, Vector::const_iterator b, size_t n);
void G(double angle, size_t row, Vector& v);
void RG(double angle, size_t row, Vector& v);
void H(Vector::const_iterator begin, Vector::const_iterator end, Vector& v);
void H(Vector::const_iterator begin, Vector::const_iterator end, Vector& v,
       size_t v_size);
size_t argmax(const Vector& a);
SVD getGivensPerfomance(const Vector& v, size_t rows, size_t cols);
Vector getHouseholderPerfomance(const Vector& v, size_t rows, size_t cols);
void vecnmult(Vector::iterator a, Vector::const_iterator b, size_t n);
void vecnmult(Vector& a, const Vector& b, size_t n);
Vector elemwisemult(const Vector& a, const Vector& b, size_t n);

}  // namespace neural_network
