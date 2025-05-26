#include "GivensLayer.h"
#include "gtest/gtest.h"
#include "HouseholderLayer.h"
#include "MatrixLayer.h"

namespace neural_network {
namespace tests {
TEST(MatrixLayer_forward, assertion1) {
    std::vector<double> w{1.5, 2.5, 3.2, 4,     5.23, 6.1,
                          7.2, 8.3, 9,   10.01, 11.5, 12};
    MatrixLayer l(In(3), Out(3), w);
    Matrix out = l.forward(Matrix{{1}, {1}, {1}});
    Matrix diff = out - Matrix{{11.2}, {26.83}, {42.51}};
    EXPECT_TRUE(1e-5 > diff.norm());
}

TEST(MatrixLayer_forwardOnTrain, assertion1) {
    std::vector<double> w{1.5, 2.5, 3.2, 4,     5.23, 6.1,
                          7.2, 8.3, 9,   10.01, 11.5, 12};
    MatrixLayer l(In(3), Out(3), w);
    Matrix out = l.forwardOnTrain(Matrix{{1}, {1}, {1}});
    Matrix diff = out - Matrix{{11.2}, {26.83}, {42.51}};
    EXPECT_TRUE(1e-5 > diff.norm());
}

TEST(MatrixLayer_sizeIn, assertion1) {
    std::vector<double> w{-1.5, -2.5, 0.12, 34, 24};
    MatrixLayer l(In(4), Out(1), w);
    ASSERT_EQ(4, l.sizeIn());
}

TEST(MatrixLayer_sizeOut, assertion1) {
    std::vector<double> w{-1.5, -2.5, 0.12, 34, 24};
    MatrixLayer l(In(4), Out(1), w);
    ASSERT_EQ(1, l.sizeOut());
}

TEST(MatrixLayer_backwardCalcGradient, assertion1) {
    std::vector<double> w{1, 2, 3, 4};
    MatrixLayer l(In(1), Out(2), w);
    Matrix u{{3}, {4}};
    Matrix z{{1}, {2}};
    Matrix x{{1}};
    Matrix g = l.backwardCalcGradient(u, x, z);
    Matrix diff_g = g - Matrix{{3, 3}, {4, 4}};
    EXPECT_TRUE(1e-5 > diff_g.norm());
}

TEST(MatrixLayer_update, assertion1) {
    std::vector<double> w{1, 2, 3, 4};
    MatrixLayer l(In(1), Out(2), w);
    l.update(Matrix{{1, 1}, {1, 1}}, 0.5);
    Matrix output = l.forward(Matrix{{1}});
    Matrix expected{{2}, {6}};
    EXPECT_TRUE(1e-5 > (output - expected).norm());
}
}  // namespace tests
}  // namespace neural_network
