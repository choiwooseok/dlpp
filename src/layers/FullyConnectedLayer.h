#pragma once

#include <cassert>
#include <cmath>

#include "base/BaseLayer.h"

class FullyConnectedLayer : public BaseLayer {
public:
  explicit FullyConnectedLayer(int numInput, int numOutput)
      : BaseLayer("FullyConnected"), numInput_(numInput), numOutput_(numOutput),
        accumSteps_(0) {
    initializeParameters();
  }

  virtual ~FullyConnectedLayer() = default;

  tensor_t forward(const tensor_t &input) override {
    // Extract dimensions
    const auto dims = extractDimensions(input);

    assert(dims.featureSize == numInput_ &&
           "FullyConnectedLayer: input feature size mismatch");

    if (dims.batchSize == 0) {
      return createOutputTensor(0);
    }

    // Create output tensor
    tensor_t output = createOutputTensor(dims.batchSize);

    // Optimized batch GEMM: Y = X * W^T + b
    forwardGEMM(input, output, dims);

    // Cache input for backward (only store reference/shape info)
    lastInputShape_ = input.shape;
    lastInput_ = input;

    return output;
  }

  tensor_t backward(const tensor_t &dY) override {
    assert(!lastInputShape_.empty() && "Backward called before forward");

    const auto dims = extractDimensions(lastInput_);

    assert(dims.featureSize == numInput_ &&
           "FullyConnectedLayer: cached input feature size mismatch");

    if (dims.batchSize == 0) {
      return tensor_t(lastInputShape_);
    }

    // Accumulate gradients
    accumulateGradients(lastInput_, dY, dims);

    // Compute input gradients
    tensor_t dX(lastInputShape_);
    backwardGEMM(dY, dX, dims);

    return dX;
  }

  void updateParams(double eta) override {
    if (accumSteps_ == 0) {
      return;
    }

    // Fused scale and update
    const val_t invSteps = val_t(1) / static_cast<val_t>(accumSteps_);
    const val_t lr = static_cast<val_t>(eta);
    const val_t scaledLR = lr * invSteps;

    W_.noalias() -= scaledLR * dW_accum_;
    B_.noalias() -= scaledLR * dB_accum_;

    resetGradientAccumulators();
  }

  void info() override {
    cout << "[" << numInput_ << " -> " << numOutput_ << "]";
  }

  // Getters and Setters
  int getNumInput() const { return numInput_; }
  int getNumOutput() const { return numOutput_; }

  const mat_t &getWeights() const { return W_; }
  void setWeights(const mat_t &weights) {
    assert(weights.rows() == numOutput_ && weights.cols() == numInput_);
    W_ = weights;
  }

  const vec_t &getBiases() const { return B_; }
  void setBiases(const vec_t &biases) {
    assert(biases.size() == numOutput_);
    B_ = biases;
  }

private:
  struct Dimensions {
    size_t batchSize;
    size_t featureSize;
  };

  Dimensions extractDimensions(const tensor_t &input) const {
    Dimensions dims;

    // Handle different input shapes
    if (input.ndim() >= 2) {
      dims.batchSize = input.shape[0];
      const size_t total = input.totalSize();
      dims.featureSize = (dims.batchSize > 0) ? (total / dims.batchSize) : 0;
    } else {
      dims.batchSize = 1;
      dims.featureSize = input.totalSize();
    }

    return dims;
  }

  tensor_t createOutputTensor(size_t batchSize) const {
    return tensor_t({batchSize, static_cast<size_t>(numOutput_)});
  }

  void forwardGEMM(const tensor_t &input, tensor_t &output,
                   const Dimensions &dims) const {
    // Map input and output to matrices
    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto Y = output.asMatrix(dims.batchSize, numOutput_);

    // Optimized GEMM: Y = X * W^T + b
    // Using noalias() to avoid temporary allocation
    Y.noalias() = X * W_.transpose();

    // Add bias to each row
    Y.rowwise() += B_.transpose();
  }

  void accumulateGradients(const tensor_t &input, const tensor_t &dY,
                           const Dimensions &dims) {
    // Map tensors to matrices
    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);

    // Accumulate weight gradients: dW = dY^T * X
    dW_accum_.noalias() += dYmat.transpose() * X;

    // Accumulate bias gradients: dB = sum(dY, axis=0)
    dB_accum_.noalias() += dYmat.colwise().sum().transpose();

    accumSteps_ += dims.batchSize;
  }

  void backwardGEMM(const tensor_t &dY, tensor_t &dX,
                    const Dimensions &dims) const {
    // Map tensors to matrices
    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);
    auto dXmat = dX.asMatrix(dims.batchSize, dims.featureSize);

    // Compute input gradients: dX = dY * W
    dXmat.noalias() = dYmat * W_;
  }

  void initializeParameters() {
    // Initialize weights and biases
    W_ = mat_t::Zero(numOutput_, numInput_);
    B_ = vec_t::Zero(numOutput_);

    // Initialize gradient accumulators
    dW_accum_ = mat_t::Zero(numOutput_, numInput_);
    dB_accum_ = vec_t::Zero(numOutput_);

    // He initialization for better convergence
    applyHeInitialization();
  }

  void applyHeInitialization() {
    const val_t scale = std::sqrt(val_t(2) / static_cast<val_t>(numInput_));

    // Vectorized initialization
    for (int i = 0; i < numOutput_; ++i) {
      for (int j = 0; j < numInput_; ++j) {
        W_(i, j) = genRandom() * scale;
      }
    }
  }

  void resetGradientAccumulators() {
    dW_accum_.setZero();
    dB_accum_.setZero();
    accumSteps_ = 0;
  }

private:
  // Layer dimensions
  int numInput_;
  int numOutput_;

  // Trainable parameters
  mat_t W_; // (numOutput x numInput)
  vec_t B_; // (numOutput)

  // Gradient accumulators for mini-batch optimization
  mat_t dW_accum_; // (numOutput x numInput)
  vec_t dB_accum_; // (numOutput)
  size_t accumSteps_;

  // Cached for backward pass
  tensor_t lastInput_;
  std::vector<size_t> lastInputShape_;
};