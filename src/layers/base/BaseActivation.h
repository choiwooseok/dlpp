#pragma once

#include "BaseLayer.h"

class BaseActivation : public BaseLayer {
 public:
  explicit BaseActivation(const string &name) : BaseLayer(name) {}
  virtual ~BaseActivation() = default;

  tensor_t forward(const tensor_t &input) override {
    this->input = input;  // cache
    return f(input);
  }

  tensor_t backward(const tensor_t &dY) override {
    assert(dY.totalSize() == input.totalSize());
    tensor_t g = df(input);    // shape: same as input
    tensor_t dX(input.shape);  // allocate result
    size_t n = input.totalSize();
    for (size_t i = 0; i < n; ++i) {
      dX[i] = dY[i] * g[i];
    }
    return dX;
  }

  // Activation layers have no trainable parameters
  void updateParams(Optimizer * /*optimizer*/) override {}

  void info() override { cout << "Activation"; }

  // activation function
  virtual tensor_t f(const tensor_t &input) = 0;

  // derivative of activation function
  virtual tensor_t df(const tensor_t &input) = 0;

 protected:
  // cache
  tensor_t input;
};