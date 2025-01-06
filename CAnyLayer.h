#pragma once

#include <vector>

#include "AnyObject.h"
#include "CustomTypes.h"

namespace NSDetail {
template <class TBase>
class IAnyLayer : public TBase {
public:
    virtual size_t sizeIn() const = 0;
    virtual size_t sizeOut() const = 0;
    virtual std::vector<double> forward(const std::vector<double>& x) const = 0;
    virtual std::vector<double> forwardOnTrain(
        const std::vector<double>& x) const = 0;
    virtual std::vector<double> backwardCalcGradient(
        std::vector<double>& u, const std::vector<double>& x,
        std::vector<double>& z) const = 0;
    virtual void update(const std::vector<double>& grad, double step) = 0;
};

template <class TBase, class TObject>
class CAnyLayerImpl : public TBase {
    // This using is for convenience only
    using CBase = TBase;

public:
    // We need to open all constructors of the base class
    using CBase::CBase;
    size_t sizeIn() const {
        return CBase::Object().sizeIn();
    }

    size_t sizeOut() const {
        return CBase::Object().sizeOut();
    }

    std::vector<double> forward(const std::vector<double>& x) const {
        return CBase::Object().forward(x);
    }

    std::vector<double> forwardOnTrain(const std::vector<double>& x) const {
        return CBase::Object().forwardOnTrain(x);
    }

    std::vector<double> backwardCalcGradient(std::vector<double>& u, const std::vector<double>& x, std::vector<double>& z) const {
        return CBase::Object().backwardCalcGradient(u, x, z);
    }
    void update(const std::vector<double>& grad, double step) {
        CBase::Object().update(grad, step);
    }
};

// A using for convenience
using CAnyLayerT = CAnyObject<IAnyLayer, CAnyLayerImpl>;
}  // namespace NSDetail

class Linear : public NSDetail::CAnyLayerT {
    // This using is for convenience only
    using CBase = NSDetail::CAnyLayerT;

public:
    // We only need to open all constructors of the base class
    using CBase::CBase;
};
