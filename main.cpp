#include <iostream>

#include "exception.h"
#include "test.h"

int main() {
    try {
        neural_network::run_all_reports();
    } catch (...) {
        neural_network::react();
    }
    return 0;
}
