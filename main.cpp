#include <iostream>

#include "test.h"
#include "exception.h"

int main() {
    try {
        neural_network::test_echo();
    } catch (...) {
        neural_network::react();
    }
    return 0;
}
