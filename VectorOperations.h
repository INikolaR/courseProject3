#pragma once

#include "CustomTypes.h"

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
void G(double angle, size_t row, Vector& v);
void RG(double angle, size_t row, Vector& v);
size_t argmax(const Vector& a);

}  // namespace neural_network
