#pragma once

#include "layers/base/BaseActivation.h"

class LReLULayer : public BaseActivation {
private:
  static constexpr float alpha = 0.01f;

public:
  explicit LReLULayer() : BaseActivation("LReLU") {}
  virtual ~LReLULayer() = default;

  vec_t f(const vec_t &x) override {
    return x.unaryExpr([](val_t e) { return e > val_t(0) ? e : alpha * e; });
  }

  vec_t df(const vec_t &x) override {
    return x.unaryExpr([](val_t e) { return e > val_t(0) ? val_t(1) : alpha; });
  }
};
