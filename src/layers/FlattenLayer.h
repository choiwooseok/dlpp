#pragma once

#include "layers/base/BaseLayer.h"

class FlattenLayer : public BaseLayer {
 public:
  explicit FlattenLayer() : BaseLayer("Flatten") {}
  virtual ~FlattenLayer() = default;

  Tensor forward(const Tensor &input) override {
    // cache input shape for backward
    lastInput = input;

    // Get batch size (first dimension)
    size_t batchSize = (input.dim() >= 2) ? input.shape(0) : 1;

    // Flatten all dimensions after batch into single feature dimension
    size_t featSize = input.size() / batchSize;

    // Output shape: {batchSize, featSize}
    Tensor out({batchSize, featSize});

    std::copy(input.data(),
              input.data() + input.size(),
              out.data());

    return out;
  }

  Tensor backward(const Tensor &dY) override {
    // retrieve cached input shape
    const auto &s = lastInput.shape();
    Tensor dX(s);

    std::copy(dY.data(),
              dY.data() + dY.size(),
              dX.data());

    return dX;
  }

  void updateParams(Optimizer * /*optimizer*/) override {}

  void info() override {}

 private:
  Tensor lastInput;
};