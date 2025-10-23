#pragma once

#include "layers/base/BaseActivation.h"

class TanhLayer : public BaseActivation {
public:
  explicit TanhLayer() : BaseActivation("Tanh") {}
  virtual ~TanhLayer() = default;

  tensor_t f(const tensor_t &x) override {
    tensor_t out(x.shape);
    size_t n = x.totalSize();
    for (size_t i = 0; i < n; ++i) {
      val_t v = x[i];
      out[i] = std::tanh(v);
    }
    return out;
  }

  tensor_t df(const tensor_t &x) override {
    tensor_t out(x.shape);
    size_t n = x.totalSize();
    for (size_t i = 0; i < n; ++i) {
      val_t v = x[i];
      val_t t = std::tanh(v);
      out[i] = val_t(1) - t * t;
    }
    return out;
  }
};
