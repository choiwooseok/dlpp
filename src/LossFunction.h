#pragma once

#include "types.h"

using namespace std;

class MSE {
public:
  static val_t f(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    val_t loss = transform_reduce(
        label.begin(), label.end(), predicted.begin(), 0.0, plus<val_t>(),
        [](const val_t &l, const val_t &r) { return (l - r) * (l - r); });

    return loss / label.size();
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    vec_t grad(label.size());
    val_t factor = val_t(2) / label.size();

    transform(
        label.begin(), label.end(), predicted.begin(), grad.begin(),
        [factor](const val_t &l, const val_t &r) { return factor * (r - l); });

    return grad;
  }
};

class BCE {
public:
  static val_t f(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    val_t loss = val_t(0);
    for (int i = 0; i < predicted.size(); i++) {
      loss += -label[i] * log(predicted[i]) -
              (val_t(1) - label[i]) * log((val_t(1) - predicted[i]));
    }
    return loss;
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    vec_t grad(predicted.size());
    for (int i = 0; i < predicted.size(); i++) {
      grad[i] = (predicted[i] - label[i]) /
                (predicted[i] * (val_t(1) - predicted[i]));
    }
    return grad;
  }
};
