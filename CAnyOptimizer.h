#pragma once

#include <vector>

#include "AnyObject.h"
#include "CustomTypes.h"

namespace NSDetail {
template <class TBase>
class IAnyOptimizer : public TBase {
public:
    virtual void update(const std::vector<std::vector<double>>& grads) = 0;
    virtual std::string describe() const = 0;
};

template <class TBase, class TObject>
class CAnyOptimizerImpl : public TBase {
    // This using is for convenience only
    using CBase = TBase;

public:
    // We need to open all constructors of the base class
    using CBase::CBase;
    void update(const std::vector<std::vector<double>>& grads) {
        CBase::Object().update(grads);
    }

    std::string describe() const {
        return CBase::Object().describe();
    }
};

// A using for convenience
using CAnyOptimizerT = CAnyObject<IAnyOptimizer, CAnyOptimizerImpl>;
}  // namespace NSDetail

class Optimizer : public NSDetail::CAnyOptimizerT {
    // This using is for convenience only
    using CBase = NSDetail::CAnyOptimizerT;

public:
    // We only need to open all constructors of the base class
    using CBase::CBase;
};
