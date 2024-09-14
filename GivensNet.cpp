#include "GivensNet.h"

#include <cassert>
#include <iostream>

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
        temp.emplace_back(1.0);
        temp = linear_it->passForward(temp);
        temp = non_linear_it->evaluate0(temp);
    }
    return temp;
}

void GivensNet::fit(const std::vector<TrainUnit>& dataset,
                    const LossFunction& loss, size_t n_of_epochs,
                    int batch_size, double step) {
    for (size_t i = 0; i < n_of_epochs; ++i) {
        trainOneEpoch(dataset, loss, batch_size, step);
    }
}

double GivensNet::loss(const std::vector<TrainUnit>& test_dataset,
                       const LossFunction& loss) const {
    assert(!test_dataset.empty() && "dataset should not be empty");
    double l = 0;
    for (const TrainUnit& unit : test_dataset) {
        l += loss.evaluate0(predict(unit.x), unit.y);
    }
    return l / static_cast<double>(test_dataset.size());
}

double GivensNet::accuracy(const std::vector<TrainUnit> test_dataset) const {
    assert(!test_dataset.empty() && "dataset should not be empty");
    double correct_answers = 0;
    for (const TrainUnit& unit : test_dataset) {
        correct_answers += unit.y[getMaxInd(predict(unit.x))];
    }
    return correct_answers / static_cast<double>(test_dataset.size());
}

void GivensNet::trainOneEpoch(const std::vector<TrainUnit>& dataset,
                              const LossFunction& loss, int batch_size,
                              double step) {
    assert(batch_size > 0);
    for (auto it = dataset.begin(); it < dataset.end(); it += batch_size) {
        auto end_of_batch =
            (it + batch_size < dataset.end() ? it + batch_size : dataset.end());
        trainOneBatch(it, end_of_batch, loss,
                      step / static_cast<double>(end_of_batch - it));
    }
}

void GivensNet::trainOneBatch(std::vector<TrainUnit>::const_iterator begin,
                              std::vector<TrainUnit>::const_iterator end,
                              const LossFunction& loss, double step) {
    if (begin == end) {
        return;
    }
    std::vector<Gradient> to_update =
        trainOneUnit(begin->x, begin->y, loss, step);
    for (auto it = begin + 1; it != end; ++it) {
        std::vector<Gradient> add_to_update =
            trainOneUnit(it->x, it->y, loss, step);
        addGradients(to_update, add_to_update);
    }
    auto it_layers = linear_layers_.begin();
    auto it_g = to_update.rbegin();
    for (; it_layers != linear_layers_.end() && it_g != to_update.rend();
         ++it_layers, ++it_g) {
        it_layers->updateAlpha(it_g->U, step);
        it_layers->updateBeta(it_g->V, step);
        it_layers->updateSigma(it_g->sigma, step);
    }
}

std::vector<Gradient> GivensNet::trainOneUnit(const Vector& x, const Vector& y,
                                              const LossFunction& loss,
                                              double step) {
    assert(x.size() == in_ && "bad input vector size");
    Vector temp;
    temp.reserve(x.size() + 1);
    std::copy(x.begin(), x.end(), back_inserter(temp));
    std::vector<Vector> non_linear_in;
    non_linear_in.reserve(non_linear_layers_.size());
    auto linear_it = linear_layers_.begin();
    auto non_linear_it = non_linear_layers_.begin();
    for (; linear_it != linear_layers_.end() &&
           non_linear_it != non_linear_layers_.end();
         ++linear_it, ++non_linear_it) {
        temp.emplace_back(1);
        temp = linear_it->passForwardWithoutShrinking(temp);
        non_linear_in.emplace_back(temp);
        temp.resize(linear_it->sizeOut());
        temp = non_linear_it->evaluate0(temp);
    }
    Vector u = loss.evaluate1(temp, y);

    std::vector<Gradient> to_update;
    to_update.reserve(linear_layers_.size());

    auto linear_layer_it = linear_layers_.rbegin();
    auto non_linear_layer_it = non_linear_layers_.rbegin();
    auto non_linear_in_it = non_linear_in.rbegin();
    for (; linear_layer_it != linear_layers_.rend() &&
           non_linear_layer_it != non_linear_layers_.rend() &&
           non_linear_in_it != non_linear_in.rend();
         ++linear_layer_it, ++non_linear_layer_it, ++non_linear_in_it) {
        u.resize(linear_layer_it->sizeOut());
        u *= non_linear_layer_it->evaluate1(*non_linear_in_it);
        Gradient g =
            linear_layer_it->passBackwardAndCalcGradient(u, *non_linear_in_it);
        to_update.emplace_back(g);
    }
    return to_update;
}

void GivensNet::addGradients(std::vector<Gradient>& a,
                             const std::vector<Gradient>& b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        a[i].U += b[i].U;
        a[i].V += b[i].V;
        a[i].sigma += b[i].sigma;
    }
}

}  // namespace neural_network
