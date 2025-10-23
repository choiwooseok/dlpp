#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "types.h"

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

    for (size_t i = 0; i < out.totalSize(); ++i) {
      out.data[i] = g(static_cast<int>(i));
    }
    return out;
  }

  static val_t f(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      cout << "label.size(): " << label.size()
           << ", predicted.size(): " << predicted.size() << endl;
      throw runtime_error("label.size() != predicted.size()");
    }

    val_t loss = transform_reduce(
        label.begin(), label.end(),
        predicted.begin(),
        val_t(0),
        plus<val_t>(),
        [](const val_t &l, const val_t &r) {
          return (l - r) * (l - r);
        });

    return loss / static_cast<val_t>(label.size());
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    vec_t grad(static_cast<int>(label.size()));
    val_t factor = val_t(2) / (val_t)label.size();

    transform(
        label.begin(), label.end(),
        predicted.begin(),
        grad.data(),
        [factor](const val_t &l, const val_t &r) {
          return factor * (r - l);
        });

    return grad;
  }
};
