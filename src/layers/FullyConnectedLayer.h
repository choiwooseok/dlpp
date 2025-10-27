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
        accumSteps_(0) {
    _initializeParameters(random);
  }

  virtual ~FullyConnectedLayer() = default;

 public:
  tensor_t forward(const tensor_t &input) override {
    // Extract dimensions
    const auto dims = _extractDims(input);

    assert(dims.featureSize == numInput_ && "FullyConnectedLayer: input feature size mismatch");

    if (dims.batchSize == 0) {
      return _createOutputTensor(0);
    }

    tensor_t output = _createOutputTensor(dims.batchSize);

    // batch GEMM: Y = X * W^T + b
    _forward(input, output, dims);

    lastInput_ = input;
    return output;
  }

  tensor_t backward(const tensor_t &dY) override {
    const auto dims = _extractDims(lastInput_);

    assert(dims.featureSize == numInput_ && "FullyConnectedLayer: cached input feature size mismatch");

    if (dims.batchSize == 0) {
      return tensor_t(lastInput_.shape);
    }

    // Accumulate gradients
    _accumulateGradients(lastInput_, dY, dims);

    // Compute input gradients
    tensor_t dX(lastInput_.shape);
    _backward(dY, dX, dims);

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
  struct Dimensions {
    size_t batchSize;
    size_t featureSize;
  };

  Dimensions _extractDims(const tensor_t &input) const {
    bool isNdimGT2 = input.ndim() >= 2;

    Dimensions dims;
    dims.batchSize = isNdimGT2 ? input.shape[0] : 1;
    dims.featureSize = isNdimGT2 ? ((dims.batchSize > 0) ? (input.totalSize() / dims.batchSize) : 0)
                                 : input.totalSize();
    return dims;
  }

  tensor_t _createOutputTensor(size_t batchSize) const {
    return tensor_t({batchSize, static_cast<size_t>(numOutput_)});
  }

  void _forward(const tensor_t &input, tensor_t &output, const Dimensions &dims) const {
    // Map input and output to matrices
    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto Y = output.asMatrix(dims.batchSize, numOutput_);

    // GEMM: Y = X * W^T + b
    // Using noalias() to avoid temporary allocation
    Y.noalias() = X * weights_.transpose();

    // Add bias to each row
    Y.rowwise() += biases_.transpose();
  }

  void _accumulateGradients(const tensor_t &input, const tensor_t &dY, const Dimensions &dims) {
    // Map tensors to matrices
    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);

    // Accumulate weight gradients: dW = dY^T * X
    dW_accum_.noalias() += dYmat.transpose() * X;

    // Accumulate bias gradients: dB = sum(dY, axis=0)
    dB_accum_.noalias() += dYmat.colwise().sum().transpose();

    accumSteps_ += dims.batchSize;
  }

  void _backward(const tensor_t &dY, tensor_t &dX, const Dimensions &dims) const {
    // Map tensors to matrices
    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);
    auto dXmat = dX.asMatrix(dims.batchSize, dims.featureSize);

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
};