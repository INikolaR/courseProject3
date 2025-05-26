#include "GivensLayer.h"
#include "gtest/gtest.h"

namespace neural_network {
namespace tests {
TEST(GivensLayer_forward, assertion1) {
    std::vector<double> w{1.5, 2.5, 3.2, 4,     5.23, 6.1,
                          7.2, 8.3, 9,   10.01, 11.5, 12};
    GivensLayer l(In(3), Out(3), w);
    Matrix out = l.forward(Matrix{{1}, {1}, {1}});
    Matrix diff = out - Matrix{{11.2}, {26.83}, {42.51}};
    EXPECT_TRUE(1e-5 > diff.norm());
}

TEST(GivensLayer_forward, assertion2) {
    std::vector<double> w{-1.5, -2.5, 0.12, 34, 24};
    GivensLayer l(In(4), Out(1), w);
    Matrix out = l.forward(Matrix{{1}, {2}, {3}, {4}});
    Matrix diff = out - Matrix{{153.86}};
    EXPECT_TRUE(1e-5 > diff.norm());
}

TEST(GivensLayer_forwardOnTrain, assertion1) {
    std::vector<double> w{-1.5, -2.5, 0.12, 34, 24};
    GivensLayer l(In(4), Out(1), w);
    Matrix out = l.forwardOnTrain(Matrix{{1}, {2}, {3}, {4}});
    Matrix diff =
        out - Matrix{{153.86}, {-1.13333}, {-2.22745}, {-2.9889}, {-1.48976}};
    EXPECT_TRUE(1e-5 > diff.norm());
}

TEST(GivensLayer_sizeIn, assertion1) {
    std::vector<double> w{-1.5, -2.5, 0.12, 34, 24};
    GivensLayer l(In(4), Out(1), w);
    ASSERT_EQ(4, l.sizeIn());
}

TEST(GivensLayer_sizeOut, assertion1) {
    std::vector<double> w{-1.5, -2.5, 0.12, 34, 24};
    GivensLayer l(In(4), Out(1), w);
    ASSERT_EQ(1, l.sizeOut());
}

TEST(GivensLayer_backwardCalcGradient, assertion1) {
    std::vector<double> w{1, 2, 3, 4};
    GivensLayer l(In(1), Out(2), w);
    Matrix u{{3}, {4}};
    Matrix z{{1}, {2}};
    Matrix x{{1}};
    Matrix g = l.backwardCalcGradient(u, x, z);
    Matrix diff_g = g - Matrix{{2}, {1.99111}, {0.324122}, {7.5}};
    EXPECT_TRUE(1e-5 > diff_g.norm());
}

TEST(GivensLayer_update, assertion1) {
    std::vector<double> w{1, 2, 3, 4};
    GivensLayer l(In(1), Out(2), w);
    l.update(Matrix{{1}, {1}, {1}, {1}}, 0.5);
    Matrix output = l.forward(Matrix{{1}});
    Matrix expected{{-0.576044}, {5.46805}};
    EXPECT_TRUE(1e-5 > (output - expected).norm());
}
}  // namespace tests
}  // namespace neural_network
