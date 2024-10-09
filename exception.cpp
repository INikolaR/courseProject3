#include "exception.h"

#include <iostream>

void neural_network::react() {
    try {
        throw;
    } catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
    } catch (...) {
        std::cout << "Unknown exception\n";
    }
}