#pragma once

#include "BaseLayer.h"

class LReLULayer : public BaseLayer {
private:
  static constexpr float alpha = 0.01f;

public:
  explicit LReLULayer(int numInput) : BaseLayer("LReLU", numInput, numInput) {
    input.resize(numInput);
    output.resize(numInput);
  };
  virtual ~LReLULayer() = default;

  vec_t forward(const vec_t &input) override {
    this->input = input;
    for (int i = 0; i < input.size(); ++i) {
      output(i) = input(i) > 0 ? input(i) : alpha * input(i);
    }
    return output;
  }

  vec_t backward(const vec_t &dY, double eta) override {
    vec_t dX(input.size());
    for (int i = 0; i < input.size(); ++i) {
      dX(i) = dY(i) * (input(i) > 0 ? val_t(1) : alpha);
    }
    return dX;
  }
};
