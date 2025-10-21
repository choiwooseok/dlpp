#pragma once

#include "base/BaseLayer.h"

class FullyConnectedLayer : public BaseLayer {
public:
  explicit FullyConnectedLayer(int numInput, int numOutput)
      : BaseLayer("FullyConnected"), numInput(numInput), numOutput(numOutput) {
    _init();
  }

  virtual ~FullyConnectedLayer() = default;

public:
  vec_t forward(const vec_t &input) override {
    this->input = input;
    return (W * input) + B;
  }

  vec_t backward(const vec_t &dY, double eta) override {
    vec_t dX = dY * W.transpose();
    tensor_t dW = dY * input.transpose();
    vec_t dB = dY;

    B -= eta * dB;
    W -= eta * dW;
    return dX;
  }

public:
  int getNumInput() const { return numInput; }
  int getNumOutput() const { return numOutput; }

  const tensor_t &getWeights() const { return W; }
  void setWeights(const tensor_t &weights) { this->W = weights; }

  const vec_t &getBiases() const { return B; }
  void setBiases(const vec_t &biases) { this->B = biases; }

private:
  void _init(bool rand = true) {
    W.resize(numOutput, numInput);
    B.resize(numOutput);

    if (rand) {
      for (int i = 0; i < numOutput; ++i) {
        for (int j = 0; j < numInput; ++j) {
          W(i, j) = genRandom();
        }
        B(i) = genRandom();
      }
    }
  }

private:
  // params
  int numInput;
  int numOutput;
  tensor_t W;
  vec_t B;

  // cache
  vec_t input;
};