#include "gtest/gtest.h"
#include "Linear.h"
#include "MatrixLayer.h"
#include "Net.h"

namespace neural_network {
namespace tests {
TEST(Net_construction_forward, assertion1) {
    Net net(Linear{MatrixLayer(In(1), Out(1), std::vector<double>{1, 1})},
            NonLinear::Id());
    Matrix out = net.predict(Matrix{{1}});
    Matrix expected{{2}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(Net_construction, assertion1) {
    Net net(Linear{MatrixLayer(In(1), Out(1), std::vector<double>{1, 1})},
            NonLinear::Id());
    EXPECT_EQ(1, net.getNumOfLayers());
}

TEST(Net_addLayer, assertion1) {
    Net net(Linear{MatrixLayer(In(1), Out(1), std::vector<double>{1, 1})},
            NonLinear::Id());
    net.addLayer(Linear{MatrixLayer(In(1), Out(1), std::vector<double>{1, 1})},
                 NonLinear::Id());
    EXPECT_EQ(2, net.getNumOfLayers());
}

TEST(Net_addLayer_forward, assertion1) {
    Net net(Linear{MatrixLayer(In(1), Out(1), std::vector<double>{1, 1})},
            NonLinear::Id());
    net.addLayer(Linear{MatrixLayer(In(1), Out(1), std::vector<double>{1, 1})},
                 NonLinear::Id());
    Matrix out = net.predict(Matrix{{1}});
    Matrix expected{{3}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}
}  // namespace tests
}  // namespace neural_network
