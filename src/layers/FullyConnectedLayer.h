#pragma once

#include "BaseLayer.h"
#include <chrono>
#include <random>

class FullyConnectedLayer : public BaseLayer {
private:
  int numInput;
  int numOutput;
  vector<vector<double>> weights;
  vector<double> biases;

public:
  FullyConnectedLayer(int numInput, int numOutput)
      : BaseLayer("FullyConnected"), numInput(numInput), numOutput(numOutput) {

    // Initialize weights and biases
    weights.resize(numOutput, vector<double>(numInput));
    biases.resize(numOutput);

    // random initialization
    std::random_device rd;
    std::mt19937 gen(
        rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < numOutput; ++i) {
      for (int j = 0; j < numInput; ++j) {
        weights[i][j] = dis(gen);
      }
      biases[i] = dis(gen);
    }
  }

  virtual ~FullyConnectedLayer() = default;

public:
  vector<double> forward(const vector<double> &input) override {
    this->input = input;
    output.resize(numOutput);

    for (int i = 0; i < numOutput; ++i) {
      output[i] = std::inner_product(weights[i].begin(), weights[i].end(),
                                     input.begin(), biases[i]);
    }
    return output;
  }

  vector<double> backward(const vector<double> &err,
                          double learningRate) override {
    vector<double> input_grad(numInput, 0.0);
    for (int i = 0; i < numInput; ++i) {
      for (int j = 0; j < numOutput; ++j) {
        input_grad[i] += weights[j][i] * err[j];
      }
    }

    for (int i = 0; i < numOutput; ++i) {
      biases[i] -= learningRate * err[i];
      for (int j = 0; j < numInput; ++j) {
        weights[i][j] -= learningRate * err[i] * input[j];
      }
    }

    return input_grad;
  }

public:
  int getNumInput() const { return numInput; }
  int getNumOutput() const { return numOutput; }

  const vector<vector<double>> &getWeights() const { return weights; }
  void setWeights(const vector<vector<double>> &weights) {
    this->weights = weights;
  }

  const vector<double> &getBiases() const { return biases; }
  void setBiases(const vector<double> &biases) { this->biases = biases; }
};