#pragma once

#include <cassert>
#include <cmath>

#include "base/BaseLayer.h"

class FullyConnectedLayer : public BaseLayer {
 public:
  explicit FullyConnectedLayer(int numInput,
                               int numOutput,
                               INIT random = INIT::XAVIER)
      : BaseLayer("FullyConnected"),
        numInput_(numInput),
        numOutput_(numOutput),
        accumSteps_(0),
        isDimInitialized_(false) {
    _initializeParameters(random);
  }

  virtual ~FullyConnectedLayer() = default;

 public:
  tensor_t forward(const tensor_t &input) override {
    if (isDimInitialized_ == false) {
      _initializeDims(input);
    }

    assert(cachedFeatureSize_ == numInput_ && "FullyConnectedLayer: input feature size mismatch");

    if (cachedBatchSize_ == 0) {
      return _createOutputTensor(0);
    }

    tensor_t output = _createOutputTensor(cachedBatchSize_);

    // Batch GEMM: Y = X * W^T + b
    _forward(input, output);

    lastInput_ = input;
    return output;
  }

  tensor_t backward(const tensor_t &dY) override {
    assert(cachedFeatureSize_ == numInput_ && "FullyConnectedLayer: cached input feature size mismatch");

    if (cachedBatchSize_ == 0) {
      return tensor_t(lastInput_.shape);
    }

    // Accumulate gradients (use cached batch size)
    _accumulateGradients(lastInput_, dY);

    // Compute input gradients
    tensor_t dX(lastInput_.shape);
    _backward(dY, dX);

    return dX;
  }

  void updateParams(Optimizer *optimizer) override {
    if (accumSteps_ == 0 || optimizer == nullptr) {
      return;
    }

    // Average the accumulated gradients
    mat_t avgDW = dW_accum_ / static_cast<val_t>(accumSteps_);
    vec_t avgDB = dB_accum_ / static_cast<val_t>(accumSteps_);

    // Use optimizer to update parameters
    optimizer->update(weights_, avgDW, layerId_ + "_W");
    optimizer->update(biases_, avgDB, layerId_ + "_B");

    dW_accum_.setZero();
    dB_accum_.setZero();
    accumSteps_ = 0;
  }

  void info() override {
    cout << "[" << numInput_ << " -> " << numOutput_ << "]";
  }

 public:
  // Getters and Setters
  int getNumInput() const { return numInput_; }
  int getNumOutput() const { return numOutput_; }

  const mat_t &getWeights() const { return weights_; }
  void setWeights(const mat_t &weights) {
    assert(weights.rows() == numOutput_ && weights.cols() == numInput_);
    weights_ = weights;
  }

  const vec_t &getBiases() const { return biases_; }
  void setBiases(const vec_t &biases) {
    assert(biases.size() == numOutput_);
    biases_ = biases;
  }

 private:
  void _initializeDims(const tensor_t &input) {
    bool isNdimGT2 = input.ndim() >= 2;
    cachedBatchSize_ = isNdimGT2 ? input.shape[0] : 1;

    cachedFeatureSize_ = isNdimGT2 ? ((cachedBatchSize_ > 0) ? (input.totalSize() / cachedBatchSize_)
                                                             : 0)
                                   : input.totalSize();

    isDimInitialized_ = true;
  }

  tensor_t _createOutputTensor(size_t batchSize) const {
    return tensor_t({batchSize, static_cast<size_t>(numOutput_)});
  }

  void _forward(const tensor_t &input, tensor_t &output) const {
    // Map input and output to matrices
    auto X = input.asMatrixConst(cachedBatchSize_, cachedFeatureSize_);
    auto Y = output.asMatrix(cachedBatchSize_, numOutput_);

    // GEMM: Y = X * W^T + b
    Y.noalias() = X * weights_.transpose();

    // Add bias to each row
    Y.rowwise() += biases_.transpose();
  }

  void _accumulateGradients(const tensor_t &input, const tensor_t &dY) {
    // Map tensors to matrices
    auto X = input.asMatrixConst(cachedBatchSize_, cachedFeatureSize_);
    auto dYmat = dY.asMatrixConst(cachedBatchSize_, numOutput_);

    // Accumulate weight gradients: dW = dY^T * X
    dW_accum_.noalias() += dYmat.transpose() * X;

    // Accumulate bias gradients: dB = sum(dY, axis=0)
    dB_accum_.noalias() += dYmat.colwise().sum().transpose();

    accumSteps_ += cachedBatchSize_;
  }

  void _backward(const tensor_t &dY, tensor_t &dX) const {
    // Map tensors to matrices
    auto dYmat = dY.asMatrixConst(cachedBatchSize_, numOutput_);
    auto dXmat = dX.asMatrix(cachedBatchSize_, cachedFeatureSize_);

    // Compute input gradients: dX = dY * W
    dXmat.noalias() = dYmat * weights_;
  }

  void _initializeParameters(INIT random) {
    // Initialize weights and biases
    weights_ = mat_t::Zero(numOutput_, numInput_);
    biases_ = vec_t::Zero(numOutput_);

    // Initialize gradient accumulators
    dW_accum_ = mat_t::Zero(numOutput_, numInput_);
    dB_accum_ = vec_t::Zero(numOutput_);

    switch (random) {
      case INIT::XAVIER: {
        const val_t scale = std::sqrt(val_t(2) / static_cast<val_t>(numInput_ + numOutput_));
        _randomWeight(scale);
        break;
      }
      case INIT::HE: {
        const val_t scale = std::sqrt(val_t(2) / static_cast<val_t>(numInput_));
        _randomWeight(scale);
        break;
      }
      case INIT::NONE:
      default:
        break;
    }
  }

  void _randomWeight(const val_t scale) {
    for (int r = 0; r < weights_.rows(); ++r) {
      for (int c = 0; c < weights_.cols(); ++c) {
        weights_(r, c) = genRandom() * scale;
      }
    }
  }

 private:
  // Layer dimensions
  int numInput_;
  int numOutput_;

  // Trainable parameters
  mat_t weights_;  // (numOutput x numInput)
  vec_t biases_;   // (numOutput)

  // Gradient accumulators for mini-batch optimization
  mat_t dW_accum_;  // (numOutput x numInput)
  vec_t dB_accum_;  // (numOutput)
  size_t accumSteps_;

  // Cached for backward pass
  tensor_t lastInput_;

  // Cached dimensions (initialized once on first forward)
  bool isDimInitialized_;
  size_t cachedBatchSize_ = 0;    // Fixed after first forward
  size_t cachedFeatureSize_ = 0;  // Fixed after first forward
};