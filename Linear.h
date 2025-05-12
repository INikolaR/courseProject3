#pragma once

#include <vector>

#include "AnyObject.h"
#include "CustomTypes.h"

namespace NSDetail {
template <class TBase>
class IAnyLayer : public TBase {
public:
    virtual Eigen::Index sizeIn() const = 0;
    virtual Eigen::Index sizeOut() const = 0;
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) const = 0;
    virtual Eigen::MatrixXd forwardOnTrain(const Eigen::MatrixXd& x) const = 0;
    virtual Eigen::MatrixXd backwardCalcGradient(Eigen::MatrixXd& u,
                                                 const Eigen::MatrixXd& x,
                                                 Eigen::MatrixXd& z) const = 0;
    virtual void update(const Eigen::MatrixXd& grad, double step) = 0;
    virtual std::string describe() const = 0;
    virtual Eigen::Index size() const = 0;
};

template <class TBase, class TObject>
class CAnyLayerImpl : public TBase {
    // This using is for convenience only
    using CBase = TBase;

public:
    // We need to open all constructors of the base class
    using CBase::CBase;
    Eigen::Index sizeIn() const {
        return CBase::Object().sizeIn();
    }

    Eigen::Index sizeOut() const {
        return CBase::Object().sizeOut();
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) const {
        return CBase::Object().forward(x);
    }

    Eigen::MatrixXd forwardOnTrain(const Eigen::MatrixXd& x) const {
        return CBase::Object().forwardOnTrain(x);
    }

    Eigen::MatrixXd backwardCalcGradient(Eigen::MatrixXd& u,
                                         const Eigen::MatrixXd& x,
                                         Eigen::MatrixXd& z) const {
        return CBase::Object().backwardCalcGradient(u, x, z);
    }
    void update(const Eigen::MatrixXd& grad, double step) {
        CBase::Object().update(grad, step);
    }

    std::string describe() const {
        return CBase::Object().describe();
    }

    Eigen::Index size() const {
        return CBase::Object().size();
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
