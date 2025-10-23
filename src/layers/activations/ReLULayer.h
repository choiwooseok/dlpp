#pragma once

#include "layers/base/BaseActivation.h"

class ReLULayer : public BaseActivation {
public:
  explicit ReLULayer() : BaseActivation("ReLU") {}
  virtual ~ReLULayer() = default;

  tensor_t f(const tensor_t &x) override {
    tensor_t out(x.shape);
    size_t n = x.totalSize();
    for (size_t i = 0; i < n; ++i) {
      val_t v = x[i];
      out[i] = (v > val_t(0) ? v : val_t(0));
    }
    return out;
  }

  tensor_t df(const tensor_t &x) override {
    tensor_t out(x.shape);
    size_t n = x.totalSize();
    for (size_t i = 0; i < n; ++i) {
      out[i] = (x[i] > val_t(0) ? val_t(1) : val_t(0));
    }
    return out;
  }
};