#include "gtest/gtest.h"
#include "NonLinear.h"

namespace neural_network {
namespace tests {
TEST(NonLinear_Id_evaluate0, assertion1) {
    NonLinear nl = NonLinear::Id();
    Matrix out = nl.evaluate0(Matrix{{2}, {2}, {2}});
    Matrix expected{{2}, {2}, {2}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(NonLinear_Id_evaluate1, assertion1) {
    NonLinear nl = NonLinear::Id();
    Matrix out = nl.evaluate1(Matrix{{2}, {2}, {2}});
    Matrix expected{{1}, {1}, {1}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(NonLinear_ReLU_evaluate0, assertion1) {
    NonLinear nl = NonLinear::ReLU();
    Matrix out = nl.evaluate0(Matrix{{2}, {-1}, {0}});
    Matrix expected{{2}, {0}, {0}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(NonLinear_ReLU_evaluate1, assertion1) {
    NonLinear nl = NonLinear::ReLU();
    Matrix out = nl.evaluate1(Matrix{{2}, {-1}, {0}});
    Matrix expected{{1}, {0}, {0}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(NonLinear_LeakyReLU_evaluate0, assertion1) {
    NonLinear nl = NonLinear::LeakyReLU();
    Matrix out = nl.evaluate0(Matrix{{2}, {-1}, {0}});
    Matrix expected{{2}, {-0.1}, {0}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(NonLinear_LeakyReLU_evaluate1, assertion1) {
    NonLinear nl = NonLinear::LeakyReLU();
    Matrix out = nl.evaluate1(Matrix{{2}, {-1}, {0.5}});
    Matrix expected{{1}, {0.1}, {1}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(NonLinear_Sigmoid_evaluate0, assertion1) {
    NonLinear nl = NonLinear::Sigmoid();
    Matrix out = nl.evaluate0(Matrix{{0}, {0}, {0}});
    Matrix expected{{0.5}, {0.5}, {0.5}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(NonLinear_Sigmoid_evaluate1, assertion1) {
    NonLinear nl = NonLinear::Sigmoid();
    Matrix out = nl.evaluate1(Matrix{{0}, {0}, {0}});
    Matrix expected{{0.25}, {0.25}, {0.25}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(NonLinear_Custom_evaluate0, assertion1) {
    NonLinear nl([](double x) { return x; }, [](double x) { return 1; },
                 "Id()");
    Matrix out = nl.evaluate0(Matrix{{2}, {2}, {2}});
    Matrix expected{{2}, {2}, {2}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

TEST(NonLinear_Custom_evaluate1, assertion1) {
    NonLinear nl([](double x) { return x; }, [](double x) { return 1; },
                 "Id()");
    Matrix out = nl.evaluate1(Matrix{{2}, {2}, {2}});
    Matrix expected{{1}, {1}, {1}};
    EXPECT_TRUE(1e-5 > (out - expected).norm());
}

}  // namespace tests
}  // namespace neural_network
