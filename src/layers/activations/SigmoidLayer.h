#pragma once

#include "layers/base/BaseActivation.h"

class SigmoidLayer : public BaseActivation {
public:
  explicit SigmoidLayer() : BaseActivation("Sigmoid") {}
  virtual ~SigmoidLayer() = default;

  vec_t f(const vec_t &x) override {
    return x.unaryExpr(
        [](val_t e) { return val_t(1) / (val_t(1) + std::exp(-e)); });
  }

  vec_t df(const vec_t &x) override {
    vec_t s = f(x);
    return s.array() * (val_t(1) - s.array());
  }
};
