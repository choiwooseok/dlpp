#pragma once

#include "types.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

using namespace std;

class MSE {
public:
  static val_t f(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    return f(l, p);
  }

  static tensor_t df(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    vec_t g = df(l, p);
    tensor_t out(predicted.shape);
    assert((size_t)g.size() == out.totalSize());
    for (size_t i = 0; i < out.totalSize(); ++i)
      out.data[i] = g((int)i);
    return out;
  }

  static val_t f(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      cout << "label.size(): " << label.size()
           << ", predicted.size(): " << predicted.size() << endl;
      throw runtime_error("label.size() != predicted.size()");
    }

    val_t loss = transform_reduce(
        label.begin(), label.end(), predicted.begin(), val_t(0), plus<val_t>(),
        [](const val_t &l, const val_t &r) { return (l - r) * (l - r); });

    return loss / (val_t)label.size();
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    vec_t grad((int)label.size());
    val_t factor = val_t(2) / (val_t)label.size();

    transform(
        label.begin(), label.end(), predicted.begin(), grad.data(),
        [factor](const val_t &l, const val_t &r) { return factor * (r - l); });

    return grad;
  }
};

class BCE {
public:
  static val_t f(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    return f(l, p);
  }

  static tensor_t df(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    vec_t g = df(l, p);
    tensor_t out(predicted.shape);
    assert((size_t)g.size() == out.totalSize());
    for (size_t i = 0; i < out.totalSize(); ++i)
      out.data[i] = g((int)i);
    return out;
  }

  static val_t f(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    val_t loss = transform_reduce(
        label.begin(), label.end(), predicted.begin(), val_t(0), plus<val_t>(),
        [](const val_t &l, const val_t &r) {
          return -l * log(r) - (val_t(1) - l) * log(val_t(1) - r);
        });
    return loss;
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    vec_t grad((int)predicted.size());
    transform(label.begin(), label.end(), predicted.begin(), grad.data(),
              [](const val_t &l, const val_t &r) {
                return (r - l) / (r * (val_t(1) - r));
              });
    return grad;
  }
};