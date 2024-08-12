#include "VectorOperations.h"

#include <cassert>
#include <cmath>

namespace neural_network {

void operator+=(Vector& a, const Vector& b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        a[i] += b[i];
    }
}

void operator-=(Vector& a, const Vector& b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        a[i] -= b[i];
    }
}

Vector operator-(const Vector& a, const Vector& b) {
    Vector sub;
    sub.reserve(std::min(a.size(), b.size()));
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        sub.emplace_back(a[i] - b[i]);
    }
    return sub;
}

void operator*=(Vector& a, const Vector& b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        a[i] = a[i] * b[i];
    }
}

void operator*=(Vector& a, double b) {
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] *= b;
    }
}

Vector operator*(const Vector& a, const Vector& b) {
    Vector mult;
    mult.reserve(std::min(a.size(), b.size()));
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        mult.emplace_back(a[i] * b[i]);
    }
    return mult;
}

Vector operator*(double l, const Vector& b) {
    Vector mult;
    mult.reserve(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
        mult.emplace_back(l * b[i]);
    }
    return mult;
}

Vector operator*(const Vector& b, double l) {
    return l * b;
}

void updateVector(Vector& v, const Vector& dv, double step) {
    assert(v.size() == dv.size());
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] -= step * dv[i];
    }
}

double dot(const Vector& a, const Vector& b) {
    double dot = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

void G(double angle, size_t row, Vector& v) {
    assert(row <= v.size() - 1);
    double t1 = v[row - 1] * std::cos(angle) - v[row] * std::sin(angle);
    double t2 = v[row - 1] * std::sin(angle) + v[row] * std::cos(angle);
    v[row - 1] = t1;
    v[row] = t2;
}

void RG(double angle, size_t row, Vector& v) {
    assert(row <= v.size() - 1);
    double t1 = v[row - 1] * std::cos(angle) + v[row] * std::sin(angle);
    double t2 = -v[row - 1] * std::sin(angle) + v[row] * std::cos(angle);
    v[row - 1] = t1;
    v[row] = t2;
}

}  // namespace neural_network
