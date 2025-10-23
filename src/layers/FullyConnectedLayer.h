#pragma once

#include <cassert>

#include "base/BaseLayer.h"

class FullyConnectedLayer : public BaseLayer {
public:
  explicit FullyConnectedLayer(int numInput, int numOutput)
      : BaseLayer("FullyConnected"), numInput(numInput), numOutput(numOutput) {
    _init();
  }

  virtual ~FullyConnectedLayer() = default;

public:
  tensor_t forward(const tensor_t &input) override {
    // cache input for backward
    lastInput = input;

    // batch / feature  (feature = totalSize / batch)
    size_t N = 1;
    if (input.ndim() >= 2)
      N = input.shape[0];
    size_t total = input.totalSize();
    size_t featSize = (N > 0) ? (total / N) : 0;

    assert((int)featSize == numInput &&
           "FullyConnectedLayer: input feature size mismatch");

    tensor_t out({N, (size_t)numOutput});
    if (N == 0 || featSize == 0)
      return out;

    // Eigen Map - batch GEMM: Y = X * W^T + b
    auto Xin = input.asMatrixConst(N, featSize);    // (N x feat)
    auto Yout = out.asMatrix(N, (size_t)numOutput); // (N x numOutput)

    Yout.noalias() = Xin * W.transpose();
    Yout.rowwise() += B.transpose();

    return out;
  }

  tensor_t backward(const tensor_t &dY) override {
    // batch / feature from cached input
    assert(!lastInput.shape.empty());
    size_t N = 1;
    if (lastInput.ndim() >= 2)
      N = lastInput.shape[0];
    size_t total = lastInput.totalSize();
    size_t featSize = (N > 0) ? (total / N) : 0;

    assert((int)featSize == numInput &&
           "FullyConnectedLayer: cached input feature size mismatch");

    tensor_t dX(lastInput.shape);
    if (N == 0 || featSize == 0)
      return dX;

    // Maps
    auto Xmat = lastInput.asMatrixConst(N, featSize);    // (N x feat)
    auto dYmat = dY.asMatrixConst(N, (size_t)numOutput); // (N x numOutput)

    // Accumulate gradients instead of immediate update
    dW_accum.noalias() += dYmat.transpose() * Xmat; // (numOutput x numInput)
    dB_accum.noalias() += dYmat.colwise().sum().transpose(); // (numOutput)
    accumSteps++;

    // dX = dY * W
    auto dXmap = dX.asMatrix(N, featSize);
    dXmap.noalias() = dYmat * W; // (N x numInput)

    return dX;
  }

  void updateParams(double eta) override {
    if (accumSteps == 0)
      return;

    val_t scale = val_t(1) / static_cast<val_t>(accumSteps);
    W.noalias() -= static_cast<val_t>(eta) * (dW_accum * scale);
    B.noalias() -= static_cast<val_t>(eta) * (dB_accum * scale);

    // Reset accumulators
    dW_accum.setZero();
    dB_accum.setZero();
    accumSteps = 0;
  }

  void info() override {
    cout << "[" << getNumInput() << " -> " << getNumOutput() << "]";
  }

public:
  int getNumInput() const { return numInput; }
  int getNumOutput() const { return numOutput; }

  const mat_t &getWeights() const { return W; }
  void setWeights(const mat_t &weights) { W = weights; }

  const vec_t &getBiases() const { return B; }
  void setBiases(const vec_t &biases) { B = biases; }

private:
  void _init() {
    W = mat_t::Zero(numOutput, numInput);
    B = vec_t::Zero(numOutput);

    // Gradient accumulators
    dW_accum = mat_t::Zero(numOutput, numInput);
    dB_accum = vec_t::Zero(numOutput);
    accumSteps = 0;

    // Xavier/He initialization for better convergence
    val_t scale = std::sqrt(val_t(2) / static_cast<val_t>(numInput));
    for (int i = 0; i < numOutput; ++i)
      for (int j = 0; j < numInput; ++j)
        W(i, j) = genRandom() * scale;
  }

private:
  int numInput;
  int numOutput;
  mat_t W;
  vec_t B;

  // Gradient accumulators for mini-batch optimization
  mat_t dW_accum;
  vec_t dB_accum;
  size_t accumSteps;

  tensor_t lastInput;
};