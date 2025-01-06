#include "Net.h"

#include <cassert>
#include <iostream>

#include "VectorOperations.h"

namespace neural_network {

Net::Net(Linear l, ActivationFunction f)
    : in_(l->sizeIn()),
      out_(l->sizeOut()),
      linear_layers_(std::list<Linear>()),
      non_linear_layers_(std::list<ActivationFunction>()) {
    linear_layers_.emplace_back(std::move(l));
    non_linear_layers_.emplace_back(std::move(f));
}

void Net::AddLayer(Linear l, ActivationFunction f) {
    assert(l->sizeIn() == out_);
    out_ = l->sizeOut();
    linear_layers_.emplace_back(std::move(l));
    non_linear_layers_.emplace_back(std::move(f));
}

Vector Net::predict(const Vector& x) const {
    auto linear_it = linear_layers_.begin();
    auto non_linear_it = non_linear_layers_.begin();
    Vector temp = (*linear_it)->forward(x);
    temp = non_linear_it->evaluate0(temp);
    ++linear_it;
    ++non_linear_it;
    for (; linear_it != linear_layers_.end() &&
           non_linear_it != non_linear_layers_.end();
         ++linear_it, ++non_linear_it) {
        temp = (*linear_it)->forward(temp);
        temp = non_linear_it->evaluate0(temp);
    }
    return temp;
}

void Net::fit(const std::vector<TrainUnit>& dataset, const LossFunction& loss,
              size_t n_of_epochs, int batch_size, double step) {
    for (size_t i = 0; i < n_of_epochs; ++i) {
        trainOneEpoch(dataset, loss, batch_size, step);
    }
}

double Net::loss(const std::vector<TrainUnit>& dataset,
                 const LossFunction& loss) const {
    assert(!dataset.empty() && "dataset should not be empty");
    double l = 0;
    for (const TrainUnit& unit : dataset) {
        l += loss.evaluate0(predict(unit.x), unit.y);
    }
    return l / static_cast<double>(dataset.size());
}

double Net::accuracy(const std::vector<TrainUnit> dataset) const {
    assert(!dataset.empty() && "dataset should not be empty");
    double correct_answers = 0;
    for (const TrainUnit& unit : dataset) {
        correct_answers += unit.y[argmax(predict(unit.x))];
    }
    return correct_answers / static_cast<double>(dataset.size());
}

Vector Net::trainOneEpochWithFrobeniusNorms(
    const std::vector<TrainUnit>& dataset, const LossFunction& loss,
    int batch_size, double step) {
    assert(batch_size > 0);
    Vector frobenius_norms(linear_layers_.size(), 0);
    for (auto it = dataset.begin(); it < dataset.end(); it += batch_size) {
        auto end_of_batch =
            (it + batch_size < dataset.end() ? it + batch_size : dataset.end());
        trainOneBatchWithAddingFrobeniusNorms(it, end_of_batch, loss, step,
                                              frobenius_norms);
    }
    return frobenius_norms;
}

void Net::trainOneEpoch(const std::vector<TrainUnit>& dataset,
                        const LossFunction& loss, int batch_size, double step) {
    assert(batch_size > 0);
    for (auto it = dataset.begin(); it < dataset.end(); it += batch_size) {
        auto end_of_batch =
            (it + batch_size < dataset.end() ? it + batch_size : dataset.end());
        trainOneBatch(it, end_of_batch, loss, step);
    }
}

void Net::trainOneBatch(std::vector<TrainUnit>::const_iterator begin,
                        std::vector<TrainUnit>::const_iterator end,
                        const LossFunction& loss, double step) {
    if (begin == end) {
        return;
    }
    std::vector<Vector> to_update = trainOneUnit(begin->x, begin->y, loss);
    for (auto it = begin + 1; it != end; ++it) {
        std::vector<Vector> add_to_update = trainOneUnit(it->x, it->y, loss);
        addGradients(to_update, add_to_update);
    }
    auto it_layers = linear_layers_.begin();
    auto it_g = to_update.rbegin();
    for (; it_layers != linear_layers_.end() && it_g != to_update.rend();
         ++it_layers, ++it_g) {
        (*it_layers)->update(*it_g, step / static_cast<double>(end - begin));
    }
}

void Net::trainOneBatchWithAddingFrobeniusNorms(
    std::vector<TrainUnit>::const_iterator begin,
    std::vector<TrainUnit>::const_iterator end, const LossFunction& loss,
    double step, Vector& frobenius_norms) {
    if (begin == end) {
        return;
    }
    std::vector<Vector> to_update = trainOneUnit(begin->x, begin->y, loss);
    for (auto it = begin + 1; it != end; ++it) {
        std::vector<Vector> add_to_update = trainOneUnit(it->x, it->y, loss);
        addGradients(to_update, add_to_update);
    }
    auto it_layers = linear_layers_.begin();
    auto it_frobenius_norms = frobenius_norms.begin();
    auto it_g = to_update.rbegin();
    for (; it_layers != linear_layers_.end() && it_g != to_update.rend();
         ++it_layers, ++it_g, ++it_frobenius_norms) {
        (*it_layers)->update(*it_g, step / static_cast<double>(end - begin));
        *it_frobenius_norms += dot(*it_g, *it_g);
    }
}

std::vector<Vector> Net::trainOneUnit(const Vector& x, const Vector& y,
                                      const LossFunction& loss) {
    assert(x.size() == in_ && "bad input vector size");
    Vector temp(x);
    std::vector<Vector> linear_in;
    std::vector<Vector> non_linear_in;
    linear_in.reserve(linear_layers_.size());
    non_linear_in.reserve(non_linear_layers_.size());
    auto linear_it = linear_layers_.begin();
    auto non_linear_it = non_linear_layers_.begin();
    for (; linear_it != linear_layers_.end() &&
           non_linear_it != non_linear_layers_.end();
         ++linear_it, ++non_linear_it) {
        linear_in.emplace_back(temp);
        temp = (*linear_it)->forwardOnTrain(temp);
        non_linear_in.emplace_back(temp);
        temp.resize((*linear_it)->sizeOut());
        temp = non_linear_it->evaluate0(temp);
    }
    Vector u = loss.evaluate1(temp, y);

    std::vector<Vector> to_update;
    to_update.reserve(linear_layers_.size());

    auto linear_layer_it = linear_layers_.rbegin();
    auto non_linear_layer_it = non_linear_layers_.rbegin();
    auto non_linear_in_it = non_linear_in.rbegin();
    auto linear_in_it = linear_in.rbegin();
    for (; linear_layer_it != linear_layers_.rend() &&
           non_linear_layer_it != non_linear_layers_.rend() &&
           non_linear_in_it != non_linear_in.rend() &&
           linear_in_it != linear_in.rend();
         ++linear_layer_it, ++non_linear_layer_it, ++non_linear_in_it,
         ++linear_in_it) {
        u.resize((*linear_layer_it)->sizeOut());
        u *= non_linear_layer_it->evaluate1(*non_linear_in_it);
        Vector g =
            (*linear_layer_it)
                ->backwardCalcGradient(u, *linear_in_it, *non_linear_in_it);
        to_update.emplace_back(g);
    }
    return to_update;
}

void Net::addGradients(std::vector<Vector>& a, const std::vector<Vector>& b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        a[i] += b[i];
    }
}

}  // namespace neural_network
