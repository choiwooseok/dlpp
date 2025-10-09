#pragma once

#include "BaseLayer.h"

class LReLULayer : public BaseLayer {
private:
  double alpha = 0.01f;

public:
  LReLULayer() : BaseLayer("LReLU"){};
  virtual ~LReLULayer() = default;

  vector<double> forward(const vector<double> &input) override {
    this->input = input;
    output.resize(input.size());
    for (int i = 0; i < input.size(); ++i) {
      output[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }
    return output;
  }

  vector<double> backward(const vector<double> &err,
                          double learningRate) override {
    vector<double> inputGrad(input.size());
    for (int i = 0; i < input.size(); ++i) {
      double lreluDerivative = input[i] > 0 ? 1.0 : alpha;
      inputGrad[i] = err[i] * lreluDerivative;
    }
    return inputGrad;
  }
};
