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

    val_t loss = transform_reduce(
        label.begin(), label.end(), predicted.begin(), 0.0, plus<val_t>(),
        [](const val_t &l, const val_t &r) {
          return -l * log(r) - (val_t(1) - l) * log(val_t(1) - r);
        });
    return loss;
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    vec_t grad(predicted.size());
    transform(label.begin(), label.end(), predicted.begin(), grad.begin(),
              [](const val_t &l, const val_t &r) {
                return (r - l) / (r * (val_t(1) - r));
              });
    return grad;
  }
};
