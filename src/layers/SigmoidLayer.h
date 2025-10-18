#pragma once

#include "BaseLayer.h"

class SigmoidLayer : public BaseLayer {
public:
  explicit SigmoidLayer(int numInput)
      : BaseLayer("SigmoidLayer", numInput, numInput) {
    input.resize(numInput);
    output.resize(numInput);
  };
  virtual ~SigmoidLayer() = default;

  vec_t forward(const vec_t &input) override {
    this->input = input;
    for (int i = 0; i < input.size(); ++i) {
      output(i) = val_t(1) / (val_t(1) + exp(-input(i)));
    }
    return output;
  }

  vec_t backward(const vec_t &dY, double eta) override {
    vec_t dX(input.size());
    for (int i = 0; i < input.size(); ++i) {
      dX(i) = dY(i) * output(i) * (val_t(1) - output(i));
    }
    return dX;
  }
};
