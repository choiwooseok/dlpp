#pragma once

#include "layers/base/BaseActivation.h"
#include <cmath>

class SigmoidLayer : public BaseActivation {
 public:
  explicit SigmoidLayer() : BaseActivation("Sigmoid") {}
  virtual ~SigmoidLayer() = default;

  tensor_t f(const tensor_t &x) override {
    tensor_t out(x.shape);
    size_t n = x.totalSize();
    for (size_t i = 0; i < n; ++i) {
      val_t v = x[i];
      out[i] = val_t(1) / (val_t(1) + std::exp(-v));
    }
    return out;
  }

  tensor_t df(const tensor_t &x) override {
    tensor_t s = f(x);
    tensor_t out(x.shape);
    size_t n = x.totalSize();
    for (size_t i = 0; i < n; ++i) {
      val_t sv = s[i];
      out[i] = sv * (val_t(1) - sv);
    }
    return out;
  }
};