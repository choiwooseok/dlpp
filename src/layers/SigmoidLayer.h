#pragma once

#include "BaseLayer.h"

class SigmoidLayer : public BaseLayer {
public:
  SigmoidLayer() : BaseLayer("SigmoidLayer"){};
  virtual ~SigmoidLayer() = default;

  vector<double> forward(const vector<double> &input) override {
    this->input = input;
    output.resize(input.size());
    for (int i = 0; i < input.size(); ++i) {
      output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
    return output;
  }

  vector<double> backward(const vector<double> &err,
                          double learningRate) override {
    vector<double> inputGrad(input.size());
    for (int i = 0; i < input.size(); ++i) {
      inputGrad[i] = err[i] * exp(input[i]) / pow((1 + exp(input[i])), 2);
    }
    return inputGrad;
  }
};
