#pragma once

#include "layers/base/BaseActivation.h"

class TanhLayer : public BaseActivation {
 public:
  explicit TanhLayer()
      : BaseActivation("Tanh") {}
  virtual ~TanhLayer() = default;

  Tensor f(const Tensor& x) override {
    Tensor out(x.shape());
    size_t n = x.size();

    ParallelUtil::p_transform(x.data(), out.data(), n, [](val_t v) { return std::tanh(v); });
    return out;
  }

  Tensor df(const Tensor& x) override {
    Tensor out(x.shape());
    size_t n = x.size();

    ParallelUtil::p_transform(x.data(), out.data(), n, [](val_t v) {
      val_t t = std::tanh(v);
      return val_t(1) - t * t;
    });
    return out;
  }
};