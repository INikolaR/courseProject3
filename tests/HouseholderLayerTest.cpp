#include "gtest/gtest.h"
#include "HouseholderLayer.h"

namespace neural_network {
namespace tests {
TEST(HouseholderLayer_forward, assertion1) {
    std::vector<double> w{1.5, 2.5, 3.2, 4,     5.23, 6.1,
                          7.2, 8.3, 9,   10.01, 11.5, 12};
    HouseholderLayer l(In(3), Out(3), w);
    Matrix out = l.forward(Matrix{{1}, {1}, {1}});
    Matrix diff = out - Matrix{{11.2}, {26.83}, {42.51}};
    EXPECT_TRUE(1e-5 > diff.norm());
}

TEST(HouseholderLayer_forwardOnTrain, assertion1) {
    std::vector<double> w{-1.5, -2.5, 0.12, 34, 24};
    HouseholderLayer l(In(4), Out(1), w);
    Matrix out = l.forwardOnTrain(Matrix{{1}, {2}, {3}, {4}});
    Matrix diff =
        out - Matrix{{153.86}, {2.15548}, {2.99254}, {1.88542}, {-0.492642}};
    EXPECT_TRUE(1e-5 > diff.norm());
}

TEST(HouseholderLayer_sizeIn, assertion1) {
    std::vector<double> w{-1.5, -2.5, 0.12, 34, 24};
    HouseholderLayer l(In(4), Out(1), w);
    ASSERT_EQ(4, l.sizeIn());
}

TEST(HouseholderLayer_sizeOut, assertion1) {
    std::vector<double> w{-1.5, -2.5, 0.12, 34, 24};
    HouseholderLayer l(In(4), Out(1), w);
    ASSERT_EQ(1, l.sizeOut());
}

TEST(HouseholderLayer_backwardCalcGradient, assertion1) {
    std::vector<double> w{1, 2, 3, 4};
    HouseholderLayer l(In(1), Out(2), w);
    Matrix u{{3}, {4}};
    Matrix z{{1}, {2}};
    Matrix x{{1}};
    Matrix g = l.backwardCalcGradient(u, x, z);
    Matrix diff_g = g - Matrix{{3.35208},  {2.18256}, {0},       {1.99111},
                               {0.324122}, {7.98937}, {4.14367}, {-0.0509255}};
    EXPECT_TRUE(1e-5 > diff_g.norm());
}

TEST(HouseholderLayer_update, assertion1) {
    std::vector<double> w{1, 2, 3, 4};
    HouseholderLayer l(In(1), Out(2), w);
    l.update(Matrix{{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}}, 0.5);
    Matrix output = l.forward(Matrix{{1}});
    Matrix expected{{1.15683}, {0.632463}};
    EXPECT_TRUE(1e-5 > (output - expected).norm());
}
}  // namespace tests
}  // namespace neural_network
