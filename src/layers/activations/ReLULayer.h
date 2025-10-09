#pragma once

#include "layers/base/BaseActivation.h"

class ReLULayer : public BaseActivation {
 public:
  explicit ReLULayer()
      : BaseActivation("ReLU") {}
  virtual ~ReLULayer() = default;

  Tensor f(const Tensor& x) override {
    Tensor out(x.shape());
    size_t n = x.size();

    ParallelUtil::p_transform(x.data(), out.data(), n, [](val_t v) { return v > val_t(0) ? v : val_t(0); });
    return out;
  }

  Tensor df(const Tensor& x) override {
    Tensor out(x.shape());
    size_t n = x.size();

    ParallelUtil::p_transform(x.data(), out.data(), n, [](val_t v) { return v > val_t(0) ? val_t(1) : val_t(0); });
    return out;
  }
};
