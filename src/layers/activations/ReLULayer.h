#pragma once

#include "layers/base/BaseActivation.h"

class ReLULayer : public BaseActivation {
public:
  explicit ReLULayer() : BaseActivation("ReLU") {}
  virtual ~ReLULayer() = default;

  vec_t f(const vec_t &x) override {
    return x.unaryExpr([](val_t e) { return e > val_t(0) ? e : val_t(0); });
  }

  vec_t df(const vec_t &x) override {
    return x.unaryExpr(
        [](val_t e) { return e > val_t(0) ? val_t(1) : val_t(0); });
  }
};
