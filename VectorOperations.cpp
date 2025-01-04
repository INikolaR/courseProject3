#include "VectorOperations.h"

#include <cassert>
#include <cmath>
#include <iostream>

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

void updateReversedVector(Vector& v, const Vector& dv, double step) {
    assert(v.size() == dv.size());
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] -= step * dv[v.size() - i - 1];
    }
}

double dot(const Vector& a, const Vector& b) {
    double dot = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

double dotn(Vector::const_iterator a, Vector::const_iterator b, size_t n) {
    double dot = 0.0;
    for (size_t i = 0; i < n; ++i, ++a, ++b) {
        dot += *a * *b;
    }
    return dot;
}

void G(double angle, size_t row, Vector& v) {
    assert(row > 0);
    assert(row <= v.size() - 1);
    double t1 = v[row - 1] * std::cos(angle) - v[row] * std::sin(angle);
    double t2 = v[row - 1] * std::sin(angle) + v[row] * std::cos(angle);
    v[row - 1] = t1;
    v[row] = t2;
}

void RG(double angle, size_t row, Vector& v) {
    assert(row > 0);
    assert(row <= v.size() - 1);
    double t1 = v[row - 1] * std::cos(angle) + v[row] * std::sin(angle);
    double t2 = -v[row - 1] * std::sin(angle) + v[row] * std::cos(angle);
    v[row - 1] = t1;
    v[row] = t2;
}

void H(Vector::const_iterator begin, Vector::const_iterator end, Vector& v) {
    H(begin, end, v, v.size());
}

void H(Vector::const_iterator begin, Vector::const_iterator end, Vector& v,
       size_t v_size) {
    double mult = 0;
    auto it = end;
    for (size_t i = v_size - 1; it != begin; --i, --it) {
        mult += v[i] * *(it - 1);
    }
    it = end;
    for (size_t i = v_size - 1; it != begin; --i, --it) {
        v[i] -= 2 * *(it - 1) * mult;
    }
}

size_t argmax(const Vector& a) {
    if (a.empty()) {
        return 0;
    }
    size_t max_index = 0;
    double max = a[0];
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i] > max) {
            max = a[i];
            max_index = i;
        }
    }
    return max_index;
}

Vector getGivensDecompose(EMatrix& m) {
    Vector svd_m;
    svd_m.reserve(m.cols() * (m.cols() - 1) / 2 +
                  m.cols() * (m.rows() - m.cols()));
    for (size_t col = 0; col < m.cols(); ++col) {
        for (size_t row = m.rows() - 1; row > col; --row) {
            double angle = atan2(-m(row, col), m(row - 1, col));
            EMatrix g{{cos(angle), -sin(angle)}, {sin(angle), cos(angle)}};
            m.block(row - 1, 0, 2, m.cols()).applyOnTheLeft(g);
            svd_m.emplace_back(angle);
        }
    }
    return svd_m;
}

void appendHouseholderDecompose(EMatrix& m, Vector& w) {
    for (size_t col = 0; col < m.cols(); ++col) {
        EVector c = m.col(col);
        c(col, 0) -= 1;
        c.normalize();
        for (size_t i = col; i < m.rows(); ++i) {
            w.emplace_back(c(i, 0));
        }
        m.applyOnTheLeft(EMatrix::Identity(c.size(), c.size()) -
                         2 * c * c.transpose());
    }
}

SVD getGivensPerfomance(const Vector& vector, size_t rows, size_t cols) {
    assert(vector.size() == rows * cols);
    EMatrix m(rows, cols);
    for (int i = 0, counter = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j, ++counter) {
            m(i, j) = vector[counter];
        }
    }
    Eigen::JacobiSVD<EMatrix> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EMatrix u = svd.matrixU();
    EMatrix v = svd.matrixV();
    EMatrix s = svd.singularValues();
    return SVD{getGivensDecompose(u), Vector(s.data(), s.data() + s.size()),
               getGivensDecompose(v)};
}

Vector getHouseholderPerfomance(const Vector& vector, size_t rows,
                                size_t cols) {
    assert(vector.size() == rows * cols);
    EMatrix m(rows, cols);
    for (int i = 0, counter = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j, ++counter) {
            m(i, j) = vector[counter];
        }
    }
    Eigen::JacobiSVD<EMatrix> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EMatrix u = svd.matrixU();
    EMatrix v = svd.matrixV();
    EMatrix s = svd.singularValues();
    Vector w;
    w.reserve(rows * cols + 2 * std::min(rows, cols));
    appendHouseholderDecompose(u, w);
    Vector ss(s.data(), s.data() + s.size());
    w.insert(w.end(), ss.begin(), ss.end());
    appendHouseholderDecompose(v, w);
    return w;
}

void vecnmult(Vector::iterator a, Vector::const_iterator b, size_t n) {
    for (size_t i = 0; i < n; ++i, ++a, ++b) {
        *a *= *b;
    }
}

void vecnmult(Vector& a, const Vector& b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] *= b[i];
    }
}

Vector elemwisemult(const Vector& a, const Vector& b, size_t n) {
    Vector mult;
    mult.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        mult.emplace_back(a[i] * b[i]);
    }
    return mult;
}

}  // namespace neural_network
