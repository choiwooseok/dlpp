#pragma once

#include "BaseLayer.h"

#include <chrono>
#include <random>

class FullyConnectedLayer : public BaseLayer {
public:
  explicit FullyConnectedLayer(int numInput, int numOutput)
      : BaseLayer("FullyConnected", numInput, numOutput) {
    input.resize(numInput);
    output.resize(numOutput);

    _init();
  }

  virtual ~FullyConnectedLayer() = default;

public:
  vec_t forward(const vec_t &input) override {
    this->input = input;
    output = (W * input) + B;
    return output;
  }

  vec_t backward(const vec_t &dY, double eta) override {
    vec_t dX = dY * W.transpose();
    B -= eta * dY;
    W -= eta * dY * input.transpose();
    return dX;
  }

public:
  const tensor_t &getWeights() const { return W; }
  void setWeights(const tensor_t &weights) { this->W = weights; }

  const vec_t &getBiases() const { return B; }
  void setBiases(const vec_t &biases) { this->B = biases; }

private:
  void _init(bool rand = true) {
    // Initialize weights and biases
    W.resize(numOutput, numInput);
    B.resize(numOutput);

    if (rand) {
      // random initialization
      std::random_device rd;
      std::mt19937 gen(
          rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
      std::uniform_real_distribution<> dis(val_t(-1), val_t(1));

      for (int i = 0; i < numOutput; ++i) {
        for (int j = 0; j < numInput; ++j) {
          W(i, j) = dis(gen);
        }
        B(i) = dis(gen);
      }
    }
  }

private:
  tensor_t W;
  vec_t B;
};