#pragma once

#include "BaseLayer.h"

#include <chrono>
#include <random>

class FullyConnectedLayer : public BaseLayer {
public:
  explicit FullyConnectedLayer(int numInput, int numOutput)
      : BaseLayer("FullyConnected", numInput, numOutput) {
    _init();
  }

  virtual ~FullyConnectedLayer() = default;

public:
  vec_t forward(const vec_t &input) override {
    this->input = input;
    for (int i = 0; i < numOutput; ++i) {
      output[i] = inner_product(W[i].begin(), W[i].end(), input.begin(), B[i]);
    }
    return output;
  }

  vec_t backward(const vec_t &dY, double eta) override {
    // dX = W^T dot dY
    vec_t dX(numInput, val_t(0));
    for (int i = 0; i < numInput; ++i) {
      for (int j = 0; j < numOutput; ++j) {
        dX[i] += W[j][i] * dY[j];
      }
    }

    for (int i = 0; i < numOutput; ++i) {
      B[i] -= eta * dY[i];
      for (int j = 0; j < numInput; ++j) {
        W[i][j] -= eta * dY[i] * input[j];
      }
    }

    return dX;
  }

public:
  const vector<vec_t> &getWeights() const { return W; }
  void setWeights(const vector<vec_t> &weights) { this->W = weights; }

  const vec_t &getBiases() const { return B; }
  void setBiases(const vec_t &biases) { this->B = biases; }

private:
  void _init(bool rand = true) {
    input.resize(numInput);
    output.resize(numOutput);

    // Initialize weights and biases
    W.resize(numOutput, vec_t(numInput, val_t(0)));
    B.resize(numOutput, val_t(0));

    if (rand) {
      // random initialization
      std::random_device rd;
      std::mt19937 gen(
          rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
      std::uniform_real_distribution<> dis(val_t(-1), val_t(1));

      for (int i = 0; i < numOutput; ++i) {
        for (int j = 0; j < numInput; ++j) {
          W[i][j] = dis(gen);
        }
        B[i] = dis(gen);
      }
    }
  }

private:
  tensor_t W;
  vec_t B;
};