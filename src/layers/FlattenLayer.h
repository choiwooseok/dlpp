#pragma once

#include "layers/base/BaseLayer.h"

class FlattenLayer : public BaseLayer {
 public:
  explicit FlattenLayer()
      : BaseLayer("Flatten") {}
  virtual ~FlattenLayer() = default;

  Tensor forward(const Tensor& input) override {
    shape_ = input.shape();

    size_t batchSize = (input.dim() >= 2) ? input.shape(0) : 1;
    size_t featSize = input.size() / batchSize;

    Tensor out = input;
    out.reshape({batchSize, featSize});  // Output shape: {batchSize, featSize}
    return out;
  }

  Tensor backward(const Tensor& dY) override {
    Tensor dX = dY;
    dX.reshape(shape_);
    return dX;
  }

  void updateParams(Optimizer* /*optimizer*/) override {}

  void info() override {}

 private:
  std::vector<size_t> shape_;
};