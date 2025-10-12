#pragma once

#include "BaseLayer.h"

class ReLULayer : public BaseLayer {
public:
  explicit ReLULayer(int numInput) : BaseLayer("ReLU", numInput, numInput) {
    input.resize(numInput);
    output.resize(numInput);
  };
  virtual ~ReLULayer() = default;

  vec_t forward(const vec_t &input) override {
    this->input = input;
    for (int i = 0; i < input.size(); ++i) {
      output[i] = max(input[i], val_t(0));
    }
    return output;
  }

  vec_t backward(const vec_t &dY, double eta) override {
    vec_t dX(input.size());
    for (int i = 0; i < input.size(); ++i) {
      dX[i] = dY[i] * (output[i] > val_t(0) ? val_t(1) : val_t(0));
    }
    return dX;
  }
};
