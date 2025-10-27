
#pragma once

#include "layers/base/BaseLayer.h"

class FlattenLayer : public BaseLayer {
 public:
  explicit FlattenLayer() : BaseLayer("Flatten") {}
  virtual ~FlattenLayer() = default;

 public:
  tensor_t forward(const tensor_t &input) override {
    // cache input shape for backward
    lastInput = input;
    size_t N = 1;
    if (input.ndim() >= 2) {
      N = input.shape[0];
    }
    size_t featSize = input.totalSize() / N;
    tensor_t out({N, featSize});
    for (size_t n = 0; n < N; ++n) {
      for (size_t f = 0; f < featSize; ++f) {
        out({n, f}) = input.data[(int)(n * featSize + f)];
      }
    }
    return out;
  }

  tensor_t backward(const tensor_t &dY) override {
    // retrieve cached input shape
    const auto &s = lastInput.shape;
    size_t N = 1;
    if (lastInput.ndim() >= 2) {
      N = lastInput.shape[0];
    }
    size_t featSize = lastInput.totalSize() / N;

    tensor_t dX(s);
    for (size_t n = 0; n < N; ++n) {
      for (size_t f = 0; f < featSize; ++f) {
        dX.data[static_cast<int>(n * featSize + f)] = dY({n, f});
      }
    }
    return dX;
  }

  void updateParams(Optimizer * /*optimizer*/) override {}

  void info() override {}

 private:
  tensor_t lastInput;
};