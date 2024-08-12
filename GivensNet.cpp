#include "GivensNet.h"

#include <cassert>

#include "VectorOperations.h"

namespace neural_network {

GivensNet::GivensNet(size_t in, size_t out, ActivationFunction f)
    : in_(in),
      out_(out),
      linear_layers_(std::list<GivensLayer>()),
      non_linear_layers_(std::list<ActivationFunction>()) {
    linear_layers_.emplace_back(GivensLayer(in, out));
    non_linear_layers_.emplace_back(std::move(f));
}

void GivensNet::AddLayer(size_t new_out, ActivationFunction f) {
    linear_layers_.emplace_back(GivensLayer(out_, new_out));
    non_linear_layers_.emplace_back(std::move(f));
    out_ = new_out;
}

Vector GivensNet::predict(const Vector& x) const {
    Vector temp;
    temp.reserve(x.size() + 1);
    std::copy(x.begin(), x.end(), back_inserter(temp));
    auto linear_it = linear_layers_.begin();
    auto non_linear_it = non_linear_layers_.begin();
    for (; linear_it != linear_layers_.end() &&
           non_linear_it != non_linear_layers_.end();
         ++linear_it, ++non_linear_it) {
        temp.emplace_back(1);
        temp = linear_it->passForward(temp);
        temp = non_linear_it->evaluate0(temp);
    }
    return temp;
}

void GivensNet::fit(const std::vector<TrainUnit>& dataset,
                    const LossFunction& loss, size_t n_of_epochs, double step) {
    for (size_t i = 0; i < n_of_epochs; ++i) {
        trainOneEpoch(dataset, loss, step);
    }
}

void GivensNet::trainOneEpoch(const std::vector<TrainUnit>& dataset,
                              const LossFunction& loss, double step) {
    for (const TrainUnit& train_unit : dataset) {
        trainOneUnit(train_unit.x, train_unit.y, loss, step);
    }
}

void GivensNet::trainOneUnit(const Vector& x, const Vector& y,
                             const LossFunction& loss, double step) {
    assert(x.size() == in_ && "bad input vector size");
    Vector temp;
    temp.reserve(x.size() + 1);
    std::copy(x.begin(), x.end(), back_inserter(temp));
    std::vector<Vector> linear_in;
    linear_in.reserve(linear_layers_.size());
    std::vector<Vector> non_linear_in;
    non_linear_in.reserve(non_linear_layers_.size());
    auto linear_it = linear_layers_.begin();
    auto non_linear_it = non_linear_layers_.begin();
    for (; linear_it != linear_layers_.end() &&
           non_linear_it != non_linear_layers_.end();
         ++linear_it, ++non_linear_it) {
        temp.emplace_back(1);
        linear_in.emplace_back(temp);
        temp = linear_it->passForwardWithoutShrinking(temp);
        non_linear_in.emplace_back(temp);
        temp.resize(linear_it->sizeOut());
        temp = non_linear_it->evaluate0(temp);
    }
    Vector u = loss.evaluate1(temp, y);
    auto linear_layer_it = linear_layers_.rbegin();
    auto non_linear_layer_it = non_linear_layers_.rbegin();
    auto linear_in_it = linear_in.rbegin();
    auto non_linear_in_it = non_linear_in.rbegin();
    for (; linear_layer_it != linear_layers_.rend() &&
           non_linear_layer_it != non_linear_layers_.rend() &&
           linear_in_it != linear_in.rend() &&
           non_linear_in_it != non_linear_in.rend();
         ++linear_layer_it, ++non_linear_layer_it, ++linear_in_it,
         ++non_linear_in_it) {
        u *= non_linear_layer_it->evaluate1(*non_linear_in_it);
        Gradient g =
            linear_layer_it->passBackwardAndCalcGradient(u, *non_linear_in_it);
        linear_layer_it->updateAlpha(g.U, step);
        linear_layer_it->updateSigma(g.sigma, step);
        linear_layer_it->updateBeta(g.V, step);
    }
}

}  // namespace neural_network
