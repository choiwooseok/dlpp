#pragma once

#include <limits>

#include "ThreadPool.h"
#include "layers/base/BaseActivation.h"

class SoftmaxLayer : public BaseActivation {
 public:
  explicit SoftmaxLayer()
      : BaseActivation("Softmax") {}
  virtual ~SoftmaxLayer() = default;

  Tensor f(const Tensor& input) override {
    const size_t batchSize = (input.dim() >= 2) ? input.shape(0) : 1;
    const size_t numClasses = input.size() / batchSize;

    Tensor output(input.shape());

    if (batchSize >= 4) {
      _softmaxParallel(input, output, batchSize, numClasses);
    } else {
      _softmaxSequential(input, output, batchSize, numClasses);
    }

    return output;
  }

  Tensor df(const Tensor& input) override {
    return f(input);
  }

  Tensor backward(const Tensor& dY) override {
    Tensor softmaxOutput = df(input);

    const size_t batchSize = (dY.dim() >= 2) ? dY.shape(0) : 1;
    const size_t numClasses = dY.size() / batchSize;

    Tensor dX(input.shape());

    if (batchSize >= 4) {
      _backwardParallel(dY, softmaxOutput, dX, batchSize, numClasses);
    } else {
      _backwardSequential(dY, softmaxOutput, dX, batchSize, numClasses);
    }

    return dX;
  }

 private:
  void _softmaxSequential(const Tensor& input, Tensor& output, size_t batchSize, size_t numClasses) const {
    for (size_t n = 0; n < batchSize; ++n) {
      softmaxSample(input, output, n, numClasses);
    }
  }

  void _softmaxParallel(const Tensor& input, Tensor& output, size_t batchSize, size_t numClasses) const {
    auto& pool = GlobalThreadPool::getInstance();
    std::vector<std::future<void>> futures;
    futures.reserve(batchSize);

    for (size_t n = 0; n < batchSize; ++n) {
      futures.push_back(
          pool.enqueue([this, &input, &output, n, numClasses]() { softmaxSample(input, output, n, numClasses); }));
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  void _backwardSequential(const Tensor& dY, const Tensor& softmaxOutput, Tensor& dX, size_t batchSize,
      size_t numClasses) const {
    for (size_t n = 0; n < batchSize; ++n) {
      backwardSample(dY, softmaxOutput, dX, n, numClasses);
    }
  }

  void _backwardParallel(const Tensor& dY, const Tensor& softmaxOutput, Tensor& dX, size_t batchSize,
      size_t numClasses) const {
    auto& pool = GlobalThreadPool::getInstance();
    std::vector<std::future<void>> futures;
    futures.reserve(batchSize);

    for (size_t n = 0; n < batchSize; ++n) {
      futures.push_back(pool.enqueue(
          [this, &dY, &softmaxOutput, &dX, n, numClasses]() { backwardSample(dY, softmaxOutput, dX, n, numClasses); }));
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  // Compute softmax for a single sample
  void softmaxSample(const Tensor& input, Tensor& output, size_t sampleIdx, size_t numClasses) const {
    const size_t offset = sampleIdx * numClasses;
    const val_t* inPtr = input.data() + offset;
    val_t* outPtr = output.data() + offset;

    // Find max for numerical stability
    val_t maxVal = *std::max_element(inPtr, inPtr + numClasses);

    // Compute exp(x - max) and sum
    std::transform(inPtr, inPtr + numClasses, outPtr, [maxVal](val_t v) { return std::exp(v - maxVal); });

    val_t sumExp = std::reduce(outPtr, outPtr + numClasses, val_t(0));

    // Normalize to get probabilities
    const val_t invSum = val_t(1) / sumExp;
    std::transform(outPtr, outPtr + numClasses, outPtr, [invSum](val_t v) { return v * invSum; });
  }

  // Compute backward pass for a single sample
  void backwardSample(const Tensor& dY, const Tensor& softmaxOutput, Tensor& dX, size_t sampleIdx,
      size_t numClasses) const {
    const size_t offset = sampleIdx * numClasses;
    const val_t* dYPtr = dY.data() + offset;
    const val_t* softmaxPtr = softmaxOutput.data() + offset;
    val_t* dXPtr = dX.data() + offset;

    // Compute dot product
    val_t dotProduct = std::transform_reduce(dYPtr, dYPtr + numClasses, softmaxPtr, val_t(0));

    // Compute gradient: dX_i = softmax_i * (dY_i - dotProduct)
    std::transform(dYPtr, dYPtr + numClasses, softmaxPtr, dXPtr,
        [dotProduct](val_t dy, val_t sm) { return sm * (dy - dotProduct); });
  }
};