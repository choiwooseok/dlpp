#pragma once

#include "layers/base/BaseActivation.h"
#include <cmath>

class LReLULayer : public BaseActivation {
 private:
  static constexpr val_t alpha = 0.01f;

 public:
  explicit LReLULayer() : BaseActivation("LReLU") {}
  virtual ~LReLULayer() = default;

  tensor_t f(const tensor_t &x) override {
    tensor_t out(x.shape);
    size_t n = x.totalSize();
    for (size_t i = 0; i < n; ++i) {
      val_t v = x[i];
      out[i] = (v > val_t(0) ? v : alpha * v);
    }
    return out;
  }

  tensor_t df(const tensor_t &x) override {
    tensor_t out(x.shape);
    size_t n = x.totalSize();
    for (size_t i = 0; i < n; ++i) {
      out[i] = (x[i] > val_t(0) ? val_t(1) : alpha);
    }
    return out;
  }
};