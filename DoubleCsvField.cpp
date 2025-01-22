#include "DoubleCsvField.h"

namespace neural_network {
void DoubleCsvField::processString(const std::string& s) {
    if (!s.empty()) {
        double d = stod(s);
        sum += d;
        min = std::min(min, d);
        max = std::max(max, d);
        ++count;
    }
}
double DoubleCsvField::evaluate(const std::string& s) {
    if (!s.empty()) {
        double d = stod(s);
        return max == min ? d : (d - min) / (max - min);
    } else {
        return count == 0 ? 0 : sum / count;
    }
}
}  // namespace neural_network
