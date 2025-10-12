#pragma once

#include "BaseLayer.h"

class ReLULayer : public BaseLayer {
public:
  ReLULayer() : BaseLayer("ReLU"){};
  virtual ~ReLULayer() = default;

  vector<double> forward(const vector<double> &input) override {
    this->input = input;
    output.resize(input.size());
    for (int i = 0; i < input.size(); ++i) {
      output[i] = max(input[i], 0.0);
    }
    return output;
  }

  vector<double> backward(const vector<double> &err,
                          double learningRate) override {
    vector<double> inputGrad(input.size());
    for (int i = 0; i < input.size(); ++i) {
      double reluDerivative = input[i] > 0.0 ? 1.0 : 0.0;
      inputGrad[i] = err[i] * reluDerivative;
    }
    return inputGrad;
  }
};
