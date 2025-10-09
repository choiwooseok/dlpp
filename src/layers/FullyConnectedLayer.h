#pragma once

#include "ThreadPool.h"
#include "base/BaseLayer.h"

class FullyConnectedLayer : public BaseLayer {
 public:
  explicit FullyConnectedLayer(int numInput, int numOutput, INIT random = INIT::XAVIER)
      : BaseLayer("FullyConnected"), numInput_(numInput), numOutput_(numOutput), accumSteps_(0) {
    _initializeParameters(random);
  }

  virtual ~FullyConnectedLayer() = default;

  Tensor forward(const Tensor& input) override {
    const auto dims = _extractDims(input);

    Tensor output = _createOutputTensor(dims.batchSize);
    _forward(input, output, dims);

    lastInput_ = input;
    return output;
  }

  Tensor backward(const Tensor& dY) override {
    const auto dims = _extractDims(lastInput_);

    _accumulateGradients(lastInput_, dY, dims);

    // Compute input gradients
    Tensor dX(lastInput_.shape());
    _backward(dY, dX, dims);

    return dX;
  }

  void updateParams(Optimizer* optimizer) override {
    if (accumSteps_ == 0 || optimizer == nullptr) {
      return;
    }

    // Average the accumulated gradients
    mat_t avgDW = dW_accum_ / static_cast<val_t>(accumSteps_);
    vec_t avgDB = dB_accum_ / static_cast<val_t>(accumSteps_);

    // Update parameters
    optimizer->update(weights_, avgDW, layerId_ + "_W");
    optimizer->update(biases_, avgDB, layerId_ + "_B");

    dW_accum_.setZero();
    dB_accum_.setZero();
    accumSteps_ = 0;
  }

  void info() override {
    std::cout << std::format("[{} -> {}]", numInput_, numOutput_);
  }

 public:
  int getNumInput() const {
    return numInput_;
  }
  int getNumOutput() const {
    return numOutput_;
  }

  mat_t getWeights() const {
    return weights_;
  }
  void setWeights(const mat_t& weights) {
    weights_ = weights;
  }

  const vec_t& getBiases() const {
    return biases_;
  }
  void setBiases(const vec_t& biases) {
    biases_ = biases;
  }

 private:
  struct Dimensions {
    size_t batchSize;
    size_t featureSize;
  };

  Dimensions _extractDims(const Tensor& input) const {
    bool isNdimGT2 = input.dim() >= 2;

    Dimensions dims;
    dims.batchSize = isNdimGT2 ? input.shape(0) : 1;
    dims.featureSize = isNdimGT2 ? ((dims.batchSize > 0) ? (input.size() / dims.batchSize) : 0) : input.size();

    if (dims.featureSize != numInput_) {
      throw LayerException(getName(),
          std::format("input feature size mismatch feat: {} numIn: {}", dims.featureSize, numInput_));
    }

    return dims;
  }

  Tensor _createOutputTensor(size_t batchSize) const {
    return Tensor({batchSize, static_cast<size_t>(numOutput_)});
  }

  void _forward(const Tensor& input, Tensor& output, const Dimensions& dims) const {
    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto Y = output.asMatrix(dims.batchSize, numOutput_);

    // seq
    if (dims.batchSize < 4) {
      Y.noalias() = X * weights_;
      Y.colwise() += biases_;
      return;
    }

    // par
    auto& pool = GlobalThreadPool::getInstance();
    const size_t numThreads = pool.getThreadCount();
    const size_t chunkSize = (dims.batchSize + numThreads - 1) / numThreads;

    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
      size_t start = t * chunkSize;
      if (start >= dims.batchSize)
        break;

      size_t end = std::min(start + chunkSize, dims.batchSize);
      size_t count = end - start;

      futures.push_back(pool.enqueue([this, &X, &Y, start, count]() {
        auto X_chunk = X.block(start, 0, count, numInput_);
        auto Y_chunk = Y.block(start, 0, count, numOutput_);

        Y_chunk.noalias() = X_chunk * weights_;
        Y_chunk.colwise() += biases_;
      }));
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  void _accumulateGradients(const Tensor& input, const Tensor& dY, const Dimensions& dims) {
    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);

    // seq
    if (dims.batchSize < 4) {
      // dW = X^T * dY
      dW_accum_.noalias() += X.transpose() * dYmat;
      dB_accum_ += dYmat.colwise().sum().transpose();

      accumSteps_ += dims.batchSize;
      return;
    }

    // par
    auto& pool = GlobalThreadPool::getInstance();
    const size_t numThreads = pool.getThreadCount();
    const size_t chunkSize = (dims.batchSize + numThreads - 1) / numThreads;

    std::vector<std::future<std::pair<mat_t, vec_t>>> futures;
    futures.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
      size_t start = t * chunkSize;
      if (start >= dims.batchSize)
        break;

      size_t end = std::min(start + chunkSize, dims.batchSize);
      size_t count = end - start;

      futures.push_back(pool.enqueue([this, &X, &dYmat, start, count]() -> std::pair<mat_t, vec_t> {
        auto X_chunk = X.block(start, 0, count, numInput_);
        auto dY_chunk = dYmat.block(start, 0, count, numOutput_);

        mat_t dW = X_chunk.transpose() * dY_chunk;
        vec_t dB = dY_chunk.colwise().sum().transpose();

        return {dW, dB};
      }));
    }

    for (auto& future : futures) {
      auto [dW, dB] = future.get();
      dW_accum_.noalias() += dW;
      dB_accum_.noalias() += dB;
    }

    accumSteps_ += dims.batchSize;
  }

  void _backward(const Tensor& dY, Tensor& dX, const Dimensions& dims) const {
    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);
    auto dXmat = dX.asMatrix(dims.batchSize, dims.featureSize);

    mat_t w_T = weights_.transpose();

    // seq
    if (dims.batchSize < 4) {
      // dX = dY * w^T
      dXmat.noalias() = dYmat * w_T;
      return;
    }

    // par
    auto& pool = GlobalThreadPool::getInstance();
    const size_t numThreads = pool.getThreadCount();
    const size_t chunkSize = (dims.batchSize + numThreads - 1) / numThreads;

    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);

    for (size_t t = 0; t < numThreads; ++t) {
      size_t start = t * chunkSize;
      if (start >= dims.batchSize)
        break;

      size_t end = std::min(start + chunkSize, dims.batchSize);
      size_t count = end - start;

      futures.push_back(pool.enqueue([&w_T, &dYmat, &dXmat, start, count, this]() {
        auto dY_chunk = dYmat.block(start, 0, count, numOutput_);
        auto dX_chunk = dXmat.block(start, 0, count, numInput_);

        dX_chunk.noalias() = dY_chunk * w_T;
      }));
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  void _initializeParameters(INIT random) {
    weights_ = mat_t::Zero(numInput_, numOutput_);
    biases_ = vec_t::Zero(numOutput_);

    dW_accum_ = mat_t::Zero(numInput_, numOutput_);
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
    std::generate(weights_.data(), weights_.data() + weights_.size(), [scale]() { return genRandom() * scale; });
  }

 private:
  // Layer dimensions
  int numInput_;
  int numOutput_;

  // Trainable parameters
  mat_t weights_;  // (numInput × numOutput)
  vec_t biases_;   // (numOutput)

  // Gradient accumulators
  mat_t dW_accum_;
  vec_t dB_accum_;
  size_t accumSteps_;

  // Cached for backward pass
  Tensor lastInput_;
};