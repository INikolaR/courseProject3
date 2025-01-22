#pragma once

#include <string>

namespace neural_network {
class DoubleCsvField {
public:
    void processString(const std::string& s);
    double evaluate(const std::string& s);

private:
    double sum;
    double min;
    double max;
    size_t count;
};
}  // namespace neural_network
