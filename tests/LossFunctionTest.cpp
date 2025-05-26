#include "gtest/gtest.h"
#include "LossFunction.h"

namespace neural_network {
namespace tests {
TEST(LossFunction_Euclid_evaluate0, assertion1) {
    LossFunction loss = LossFunction::Euclid();
    double diff =
        3.0 - loss.evaluate0(Matrix{{1}, {1}, {1}}, Matrix{{2}, {2}, {2}});
    EXPECT_TRUE(1e-5 > diff);
}

TEST(LossFunction_Euclid_evaluate1, assertion1) {
    LossFunction loss = LossFunction::Euclid();
    Matrix diff = Matrix{{-2}, {-2}, {-2}} -
                  loss.evaluate1(Matrix{{1}, {1}, {1}}, Matrix{{2}, {2}, {2}});
    EXPECT_TRUE(1e-5 > diff.norm());
}

TEST(LossFunction_Manhattan_evaluate0, assertion1) {
    LossFunction loss = LossFunction::Manhattan();
    double diff =
        3.0 - loss.evaluate0(Matrix{{1}, {1}, {1}}, Matrix{{2}, {2}, {2}});
    EXPECT_TRUE(1e-5 > diff);
}

TEST(LossFunction_Manhattan_evaluate1, assertion1) {
    LossFunction loss = LossFunction::Manhattan();
    Matrix diff = Matrix{{-1}, {-1}, {-1}} -
                  loss.evaluate1(Matrix{{1}, {1}, {1}}, Matrix{{2}, {2}, {2}});
    EXPECT_TRUE(1e-5 > diff.norm());
}

TEST(LossFunction_Custom_evaluate0, assertion1) {
    LossFunction loss(
        [](const Matrix& x, const Matrix& y) {
            Matrix d = x - y;
            return (x - y).squaredNorm();
        },
        [](const Matrix& x, const Matrix& y) { return 2 * (x - y); });
    double diff =
        3.0 - loss.evaluate0(Matrix{{1}, {1}, {1}}, Matrix{{2}, {2}, {2}});
    EXPECT_TRUE(1e-5 > diff);
}

TEST(LossFunction_Custom_evaluate1, assertion1) {
    LossFunction loss(
        [](const Matrix& x, const Matrix& y) {
            return (x - y).cwiseAbs().sum();
        },
        [](const Matrix& x, const Matrix& y) { return (x - y).cwiseSign(); });
    Matrix diff = Matrix{{-1}, {-1}, {-1}} -
                  loss.evaluate1(Matrix{{1}, {1}, {1}}, Matrix{{2}, {2}, {2}});
    EXPECT_TRUE(1e-5 > diff.norm());
}

}  // namespace tests
}  // namespace neural_network
