#pragma once

#include "BaseLayer.h"
#include "ThreadPool.h"

class BaseActivation : public BaseLayer {
 public:
  explicit BaseActivation(const std::string& name)
      : BaseLayer(name) {}
  virtual ~BaseActivation() = default;

  Tensor forward(const Tensor& input) override {
    this->input = input;  // cache
    return f(input);
  }

  Tensor backward(const Tensor& dY) override {
    Tensor g = df(input);
    Tensor dX(input.shape());
    size_t n = input.size();

    ParallelUtil::p_transform(dY.data(), g.data(), dX.data(), n,
        [](val_t dy, val_t gradient) { return dy * gradient; });
    return dX;
  }

  // Activation layers have no trainable parameters
  void updateParams(Optimizer* /*optimizer*/) override {}

  void info() override {
    std::cout << "Activation";
  }

  // activation function
  virtual Tensor f(const Tensor& input) = 0;

  // derivative of activation function
  virtual Tensor df(const Tensor& input) = 0;

 protected:
  // cache
  Tensor input;
};