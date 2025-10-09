#pragma once

#include "base/BaseLayer.h"
#include "ThreadPool.h"

class FullyConnectedLayer : public BaseLayer {
 public:
  explicit FullyConnectedLayer(int numInput,
                               int numOutput,
                               INIT random = INIT::XAVIER,
                               bool enableParallel = true)
      : BaseLayer("FullyConnected"),
        numInput_(numInput),
        numOutput_(numOutput),
        accumSteps_(0),
        enableParallel_(enableParallel) {
    _initializeParameters(random);
  }

  virtual ~FullyConnectedLayer() = default;

  Tensor forward(const Tensor &input) override {
    const auto dims = _extractDims(input);
    if (dims.featureSize != numInput_) {
      throw LayerException(getName(), std::format("input feature size mismatch feat: {} numIn: {}", dims.featureSize, numInput_));
    }

    if (dims.batchSize == 0) {
      return _createOutputTensor(0);
    }

    Tensor output = _createOutputTensor(dims.batchSize);

    if (enableParallel_ && dims.batchSize >= 4) {
      _forwardParallel(input, output, dims);
    } else {
      _forward(input, output, dims);
    }

    lastInput_ = input;
    return output;
  }

  Tensor backward(const Tensor &dY) override {
    const auto dims = _extractDims(lastInput_);

    if (dims.featureSize != numInput_) {
      throw LayerException(getName(), std::format("input feature size mismatch feat: {} numIn: {}", dims.featureSize, numInput_));
    }

    if (dims.batchSize == 0) {
      return Tensor(lastInput_.shape());
    }

    if (enableParallel_ && dims.batchSize >= 4) {
      _accumulateGradientsParallel(lastInput_, dY, dims);
    } else {
      _accumulateGradients(lastInput_, dY, dims);
    }

    // Compute input gradients
    Tensor dX(lastInput_.shape());
    if (enableParallel_ && dims.batchSize >= 4) {
      _backwardParallel(dY, dX, dims);
    } else {
      _backward(dY, dX, dims);
    }

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
    std::cout << std::format("[{} -> {} parallel={}]",
                             numInput_, numOutput_,
                             enableParallel_ ? "on" : "off");
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

  void setParallelEnabled(bool enabled) { enableParallel_ = enabled; }
  bool isParallelEnabled() const { return enableParallel_; }

 private:
  struct Dimensions {
    size_t batchSize;
    size_t featureSize;
  };

  Dimensions _extractDims(const Tensor &input) const {
    bool isNdimGT2 = input.dim() >= 2;

    Dimensions dims;
    dims.batchSize = isNdimGT2 ? input.shape(0) : 1;
    dims.featureSize = isNdimGT2 ? ((dims.batchSize > 0) ? (input.size() / dims.batchSize) : 0)
                                 : input.size();
    return dims;
  }

  Tensor _createOutputTensor(size_t batchSize) const {
    return Tensor({batchSize, static_cast<size_t>(numOutput_)});
  }

  void _forward(const Tensor &input, Tensor &output, const Dimensions &dims) const {
    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto Y = output.asMatrix(dims.batchSize, numOutput_);

    Y.noalias() = X * weights_.transpose();
    Y.rowwise() += biases_.transpose();
  }

  void _forwardParallel(const Tensor &input, Tensor &output, const Dimensions &dims) const {
    auto &pool = GlobalThreadPool::getInstance();
    const size_t numThreads = pool.getThreadCount();
    const size_t chunkSize = (dims.batchSize + numThreads - 1) / numThreads;

    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);

    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto Y = output.asMatrix(dims.batchSize, numOutput_);

    for (size_t t = 0; t < numThreads; ++t) {
      size_t start = t * chunkSize;
      if (start >= dims.batchSize) break;

      size_t end = std::min(start + chunkSize, dims.batchSize);
      size_t count = end - start;

      futures.push_back(pool.enqueue([this, &X, &Y, start, count]() {
        auto X_chunk = X.block(start, 0, count, numInput_);
        auto Y_chunk = Y.block(start, 0, count, numOutput_);

        Y_chunk.noalias() = X_chunk * weights_.transpose();
        Y_chunk.rowwise() += biases_.transpose();
      }));
    }

    for (auto &future : futures) {
      future.get();
    }
  }

  void _accumulateGradients(const Tensor &input, const Tensor &dY, const Dimensions &dims) {
    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);

    dW_accum_.noalias() += dYmat.transpose() * X;
    dB_accum_.noalias() += dYmat.colwise().sum().transpose();

    accumSteps_ += dims.batchSize;
  }

  void _accumulateGradientsParallel(const Tensor &input, const Tensor &dY, const Dimensions &dims) {
    auto &pool = GlobalThreadPool::getInstance();
    const size_t numThreads = pool.getThreadCount();
    const size_t chunkSize = (dims.batchSize + numThreads - 1) / numThreads;

    std::vector<std::future<std::pair<mat_t, vec_t>>> futures;
    futures.reserve(numThreads);

    auto X = input.asMatrixConst(dims.batchSize, dims.featureSize);
    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);

    for (size_t t = 0; t < numThreads; ++t) {
      size_t start = t * chunkSize;
      if (start >= dims.batchSize) break;

      size_t end = std::min(start + chunkSize, dims.batchSize);
      size_t count = end - start;

      futures.push_back(pool.enqueue([this, &X, &dYmat, start, count]()
                                         -> std::pair<mat_t, vec_t> {
        auto X_chunk = X.block(start, 0, count, numInput_);
        auto dY_chunk = dYmat.block(start, 0, count, numOutput_);

        mat_t dW = dY_chunk.transpose() * X_chunk;
        vec_t dB = dY_chunk.colwise().sum().transpose();

        return {dW, dB};
      }));
    }

    for (auto &future : futures) {
      auto [dW, dB] = future.get();
      dW_accum_.noalias() += dW;
      dB_accum_.noalias() += dB;
    }

    accumSteps_ += dims.batchSize;
  }

  // 순차 backward
  void _backward(const Tensor &dY, Tensor &dX, const Dimensions &dims) const {
    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);
    auto dXmat = dX.asMatrix(dims.batchSize, dims.featureSize);

    dXmat.noalias() = dYmat * weights_;
  }

  // 병렬 backward
  void _backwardParallel(const Tensor &dY, Tensor &dX, const Dimensions &dims) const {
    auto &pool = GlobalThreadPool::getInstance();
    const size_t numThreads = pool.getThreadCount();
    const size_t chunkSize = (dims.batchSize + numThreads - 1) / numThreads;

    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);

    auto dYmat = dY.asMatrixConst(dims.batchSize, numOutput_);
    auto dXmat = dX.asMatrix(dims.batchSize, dims.featureSize);

    for (size_t t = 0; t < numThreads; ++t) {
      size_t start = t * chunkSize;
      if (start >= dims.batchSize) break;

      size_t end = std::min(start + chunkSize, dims.batchSize);
      size_t count = end - start;

      futures.push_back(pool.enqueue([this, &dYmat, &dXmat, start, count]() {
        auto dY_chunk = dYmat.block(start, 0, count, numOutput_);
        auto dX_chunk = dXmat.block(start, 0, count, numInput_);

        dX_chunk.noalias() = dY_chunk * weights_;
      }));
    }

    for (auto &future : futures) {
      future.get();
    }
  }

  void _initializeParameters(INIT random) {
    weights_ = mat_t::Zero(numOutput_, numInput_);
    biases_ = vec_t::Zero(numOutput_);

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
    const int totalElements = weights_.rows() * weights_.cols();
    val_t *data = weights_.data();

    std::generate(data, data + totalElements,
                  [scale]() { return genRandom() * scale; });
  }

 private:
  // Layer dimensions
  int numInput_;
  int numOutput_;
  bool enableParallel_;

  // Trainable parameters
  mat_t weights_;  // (numOutput x numInput)
  vec_t biases_;   // (numOutput)

  // Gradient accumulators for mini-batch optimization
  mat_t dW_accum_;  // (numOutput x numInput)
  vec_t dB_accum_;  // (numOutput)
  size_t accumSteps_;

  // Cached for backward pass
  Tensor lastInput_;
};