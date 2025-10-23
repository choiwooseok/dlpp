#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "layers/base/BaseActivation.h"

class SoftmaxLayer : public BaseActivation {
 public:
  explicit SoftmaxLayer() : BaseActivation("Softmax") {}
  virtual ~SoftmaxLayer() = default;

  // Softmax activation function
  tensor_t f(const tensor_t &input) override {
    // Softmax is applied per sample (row-wise for batch)
    // Shape: (N, C) where N=batch, C=classes

    const size_t batchSize = (input.ndim() >= 2) ? input.shape[0] : 1;
    const size_t numClasses = input.totalSize() / batchSize;

    tensor_t output(input.shape);

    // Process each sample in batch
    for (size_t n = 0; n < batchSize; ++n) {
      softmaxSample(input, output, n, numClasses);
    }

    return output;
  }

  // Derivative of softmax
  // For softmax, the derivative is complex because output depends on all inputs
  // We return the output itself here and handle the full Jacobian in backward
  tensor_t df(const tensor_t &input) override {
    // For softmax, df is not element-wise
    // We compute softmax output and handle Jacobian in backward()
    return f(input);
  }

  // Override backward to handle softmax Jacobian properly
  tensor_t backward(const tensor_t &dY) override {
    // Softmax Jacobian: J_ij = s_i * (δ_ij - s_j)
    // Gradient: dL/dx_i = s_i * (dL/dy_i - Σ_j(dL/dy_j * s_j))

    assert(dY.totalSize() == input.totalSize());

    // df() returns softmax output
    tensor_t softmaxOutput = df(input);

    const size_t batchSize = (dY.ndim() >= 2) ? dY.shape[0] : 1;
    const size_t numClasses = dY.totalSize() / batchSize;

    tensor_t dX(input.shape);

    // Process each sample in batch
    for (size_t n = 0; n < batchSize; ++n) {
      backwardSample(dY, softmaxOutput, dX, n, numClasses);
    }

    return dX;
  }

 private:
  // Compute softmax for a single sample
  void softmaxSample(const tensor_t &input, tensor_t &output, size_t sampleIdx,
                     size_t numClasses) const {
    const size_t offset = sampleIdx * numClasses;

    // Find max for numerical stability: softmax(x) = softmax(x - max(x))
    val_t maxVal = -std::numeric_limits<val_t>::infinity();
    for (size_t i = 0; i < numClasses; ++i) {
      maxVal = std::max(maxVal, input[offset + i]);
    }

    // Compute exp(x - max) and sum
    val_t sumExp = 0.0;
    for (size_t i = 0; i < numClasses; ++i) {
      const val_t expVal = std::exp(input[offset + i] - maxVal);
      output[offset + i] = expVal;
      sumExp += expVal;
    }

    // Normalize to get probabilities
    const val_t invSum = val_t(1) / sumExp;
    for (size_t i = 0; i < numClasses; ++i) {
      output[offset + i] *= invSum;
    }
  }

  // Compute backward pass for a single sample
  // Jacobian of softmax: J_ij = s_i * (δ_ij - s_j)
  // Gradient: dL/dx_i = s_i * (dL/dy_i - Σ_j(dL/dy_j * s_j))
  void backwardSample(const tensor_t &dY, const tensor_t &softmaxOutput,
                      tensor_t &dX, size_t sampleIdx, size_t numClasses) const {
    const size_t offset = sampleIdx * numClasses;

    // Compute dot product: Σ_j (dY_j * softmax_j)
    val_t dotProduct = 0.0;
    for (size_t j = 0; j < numClasses; ++j) {
      dotProduct += dY[offset + j] * softmaxOutput[offset + j];
    }

    // Compute gradient: dX_i = softmax_i * (dY_i - dotProduct)
    for (size_t i = 0; i < numClasses; ++i) {
      dX[offset + i] =
          softmaxOutput[offset + i] * (dY[offset + i] - dotProduct);
    }
  }
};