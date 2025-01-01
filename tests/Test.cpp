#include "GivensLayer.h"
#include "gtest/gtest.h"
#include "MatrixLayer.h"

namespace neural_network {
namespace tests {
TEST(GivensLayer_forward, assertion1) {
    Vector w = {1.5, 2.5, 3.2, 4, 5.23, 6.1, 7.2, 8.3, 9, 10.01, 11.5, 12};
    GivensLayer l(w, 3, 3);
    Vector out = l.forward({1, 1, 1});
    Vector diff = out - Vector{11.2, 26.83, 42.51};
    EXPECT_TRUE(1e-5 > dot(diff, diff));
}
TEST(GivensLayer_forward, assertion2) {
    Vector w = {-1.5, -2.5, 0.12, 34, 24};
    GivensLayer l(w, 4, 1);
    Vector out = l.forward({1, 2, 3, 4});
    Vector diff = out - Vector{153.86};
    EXPECT_TRUE(1e-5 > dot(diff, diff));
}
TEST(GivensLayer_forwardWithoutShrinking, assertion1) {
    Vector w = {-1.5, -2.5, 0.12, 34, 24};
    GivensLayer l(w, 4, 1);
    Vector out = l.forwardOnTrain({1, 2, 3, 4});
    Vector diff = out - Vector{153.86, -1.13333, -2.22745, -2.9889, -1.48976};
    EXPECT_TRUE(1e-5 > dot(diff, diff));
}
TEST(GivensLayer_sizeIn, assertion1) {
    Vector w = {-1.5, -2.5, 0.12, 34, 24};
    GivensLayer l(w, 4, 1);
    ASSERT_EQ(4, l.sizeIn());
}
TEST(GivensLayer_sizeOut, assertion1) {
    Vector w = {-1.5, -2.5, 0.12, 34, 24};
    GivensLayer l(w, 4, 1);
    ASSERT_EQ(1, l.sizeOut());
}
TEST(GivensLayer_update, assertion1) {
    Vector w = {1, 2, 3, 4};
    GivensLayer l(w, 1, 2);
    l.update({1, 0.5, 0.4, 0.1}, 2);
    Vector out = l.forward({1});
    Vector diff = out - Vector{-5.88036, -0.29761};
    EXPECT_TRUE(1e-5 > dot(diff, diff));
}
TEST(GivensLayer_backwardCalcGradient, assertion1) {
    Vector w = {1, 2, 3, 4};
    GivensLayer l(w, 1, 2);
    Vector u = {3, 4};
    Vector z = {1, 2};
    Vector x = {1, 2};
    Vector g = l.backwardCalcGradient(u, x, z);
    Vector diff_g = g - Vector{2, 1.99111, 0.324122, 7.5};
    Vector diff_cumulative_u = u - Vector{15.6733, 21.5255};
    Vector diff_cumulative_z = z - Vector{0.470871, 0.168168};
    EXPECT_TRUE(1e-5 > dot(diff_g, diff_g));
    EXPECT_TRUE(1e-5 > dot(diff_cumulative_u, diff_cumulative_u));
    EXPECT_TRUE(1e-5 > dot(diff_cumulative_z, diff_cumulative_z));
}

TEST(MatrixLayer_forward, assertion1) {
    Vector w = {1.5, 2.5, 3.2, 4, 5.23, 6.1, 7.2, 8.3, 9, 10.01, 11.5, 12};
    MatrixLayer l(w, 3, 3);
    Vector out = l.forward({1, 1, 1});
    Vector diff = out - Vector{11.2, 26.83, 42.51};
    EXPECT_TRUE(1e-5 > dot(diff, diff));
}
}  // namespace tests
}  // namespace neural_network

int main(int nArgs, char** vArgs) {
    ::testing::InitGoogleTest(&nArgs, vArgs);
    return RUN_ALL_TESTS();
}
