#pragma once

#include "layers/base/BaseActivation.h"

class SigmoidLayer : public BaseActivation {
 public:
  explicit SigmoidLayer()
      : BaseActivation("Sigmoid") {}
  virtual ~SigmoidLayer() = default;

  Tensor f(const Tensor& x) override {
    Tensor out(x.shape());
    size_t n = x.size();

    ParallelUtil::p_transform(x.data(), out.data(), n, [](val_t v) { return val_t(1) / (val_t(1) + std::exp(-v)); });
    return out;
  }

  Tensor df(const Tensor& x) override {
    Tensor s = f(x);
    Tensor out(s.shape());
    size_t n = s.size();

    ParallelUtil::p_transform(s.data(), out.data(), n, [](val_t sv) { return sv * (val_t(1) - sv); });
    return out;
  }
};
