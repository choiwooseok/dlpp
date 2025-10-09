#pragma once

#include "layers/base/BaseActivation.h"

class LReLULayer : public BaseActivation {
 private:
  static constexpr val_t alpha = 0.01f;

 public:
  explicit LReLULayer()
      : BaseActivation("LReLU") {}
  virtual ~LReLULayer() = default;

  Tensor f(const Tensor& x) override {
    Tensor out(x.shape());
    size_t n = x.size();

    ParallelUtil::p_transform(x.data(), out.data(), n, [](val_t v) { return v > val_t(0) ? v : alpha * v; });
    return out;
  }

  Tensor df(const Tensor& x) override {
    Tensor out(x.shape());
    size_t n = x.size();

    ParallelUtil::p_transform(x.data(), out.data(), n, [](val_t v) { return v > val_t(0) ? val_t(1) : alpha; });
    return out;
  }
};