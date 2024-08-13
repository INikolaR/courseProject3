#include <iostream>

#include "test.h"
#include "exception.h"

int main() {
    try {
        neural_network::run_all_tests();
    } catch (...) {
        neural_network::react();
    }
    return 0;
}
