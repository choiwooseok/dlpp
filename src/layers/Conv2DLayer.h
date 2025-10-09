#pragma once

#include "ConvUtil.h"
#include "ThreadPool.h"
#include "layers/base/BaseLayer.h"

class Conv2DLayer : public BaseLayer {
 public:
  Conv2DLayer(int inChannels, int outChannels, int kernelHeight, int kernelWidth, int stride = 1, int pad = 0,
      INIT random = INIT::XAVIER)
      : BaseLayer("Conv2D"),
        inChannels_(inChannels),
        outChannels_(outChannels),
        kernelHeight_(kernelHeight),
        kernelWidth_(kernelWidth),
        stride_(stride),
        pad_(pad),
        patchSize_(inChannels * kernelHeight * kernelWidth),
        accumSteps_(0) {
    _initializeParameters(random);
  }

  virtual ~Conv2DLayer() = default;

  Tensor forward(const Tensor& X) override {
    if (X.dim() != 4) {
      throw LayerException(getName(), std::format("Expected 4D input, got {}D", X.dim()));
    }

    if (X.shape(1) != inChannels_) {
      throw LayerException(getName(), std::format("input channel size mismatch X.shape(1): {} inChannels_: {}",
                                          X.shape(1), inChannels_));
    }

    dimCache_ = ConvUtil::getDimensions(X, kernelHeight_, kernelWidth_, stride_, pad_);

    _ensureWorkspaceSize();

    if (dimCache_.batchSize >= 4) {
      _batchedIm2colParallel(X);
    } else {
      _batchedIm2col(X);
    }

    // Batched convolution via GEMM: Y = W * X_col + b
    outWorkspace_.noalias() = weights_ * colWorkspace_;
    outWorkspace_.colwise() += biases_;

    // Convert workspace to output tensor
    Tensor Y = _workspaceToTensor();

    inputCache_ = X;
    return Y;
  }

  Tensor backward(const Tensor& dY) override {
    _ensureWorkspaceSize();

    if (dimCache_.batchSize >= 4) {
      _batchedIm2colParallel(inputCache_);
    } else {
      _batchedIm2col(inputCache_);
    }

    // Convert output gradients to matrix format
    _tensorToWorkspace(dY);

    // Compute parameter gradients via GEMM
    dW_accum_.noalias() += outWorkspace_ * colWorkspace_.transpose();
    dB_accum_.noalias() += outWorkspace_.rowwise().sum();
    accumSteps_ += dimCache_.batchSize;

    // Compute input gradients: dX_col = W^T * dY
    colWorkspace_.noalias() = weights_.transpose() * outWorkspace_;

    // col2im: scatter gradients back to input space
    Tensor dX;
    if (dimCache_.batchSize >= 4) {
      dX = _batchedCol2imParallel();
    } else {
      dX = _batchedCol2im();
    }

    return dX;
  }

  void updateParams(Optimizer* optimizer) override {
    if (accumSteps_ == 0 || optimizer == nullptr) {
      return;
    }

    // Average accumulated gradients
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
    std::cout << std::format("[inC={} outC={} k={}x{} s={} p={}]", inChannels_, outChannels_, kernelHeight_,
        kernelWidth_, stride_, pad_);
  }

 public:
  // Getters
  int getInChannels() const {
    return inChannels_;
  }
  int getOutChannels() const {
    return outChannels_;
  }
  int getKernelHeight() const {
    return kernelHeight_;
  }
  int getKernelWidth() const {
    return kernelWidth_;
  }
  int getStride() const {
    return stride_;
  }
  int getPadding() const {
    return pad_;
  }

  const mat_t& getWeights() const {
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
  Tensor _createOutputTensor() const {
    return Tensor({static_cast<size_t>(dimCache_.batchSize), static_cast<size_t>(outChannels_),
        static_cast<size_t>(dimCache_.outputHeight), static_cast<size_t>(dimCache_.outputWidth)});
  }

  void _ensureWorkspaceSize() {
    const int totalCols = dimCache_.batchSize * dimCache_.numColumns;

    if (colWorkspace_.rows() != patchSize_ || colWorkspace_.cols() != totalCols) {
      colWorkspace_.resize(patchSize_, totalCols);
    }
    if (outWorkspace_.rows() != outChannels_ || outWorkspace_.cols() != totalCols) {
      outWorkspace_.resize(outChannels_, totalCols);
    }
  }

  void _batchedIm2col(const Tensor& X) {
    for (int n = 0; n < dimCache_.batchSize; ++n) {
      const int colOffset = n * dimCache_.numColumns;
      mat_t cols = ConvUtil::im2col(X, n, dimCache_);
      colWorkspace_.block(0, colOffset, patchSize_, dimCache_.numColumns) = cols;
    }
  }

  void _batchedIm2colParallel(const Tensor& X) {
    auto& pool = GlobalThreadPool::getInstance();
    std::vector<std::future<mat_t>> futures;
    futures.reserve(dimCache_.batchSize);

    // parallel proc samples
    for (int n = 0; n < dimCache_.batchSize; ++n) {
      futures.push_back(pool.enqueue([&X, n, this]() { return ConvUtil::im2col(X, n, dimCache_); }));
    }

    // result collect
    for (int n = 0; n < dimCache_.batchSize; ++n) {
      const int colOffset = n * dimCache_.numColumns;
      mat_t cols = futures[n].get();
      colWorkspace_.block(0, colOffset, patchSize_, dimCache_.numColumns) = cols;
    }
  }

  Tensor _batchedCol2im() const {
    Tensor dX({static_cast<size_t>(dimCache_.batchSize), static_cast<size_t>(dimCache_.channels),
        static_cast<size_t>(dimCache_.inputHeight), static_cast<size_t>(dimCache_.inputWidth)});
    dX.fill(0);

    for (int n = 0; n < dimCache_.batchSize; ++n) {
      const int colOffset = n * dimCache_.numColumns;
      mat_t cols = colWorkspace_.block(0, colOffset, patchSize_, dimCache_.numColumns);
      ConvUtil::col2im(dX, cols, n, dimCache_);
    }

    return dX;
  }

  Tensor _batchedCol2imParallel() const {
    Tensor dX({static_cast<size_t>(dimCache_.batchSize), static_cast<size_t>(dimCache_.channels),
        static_cast<size_t>(dimCache_.inputHeight), static_cast<size_t>(dimCache_.inputWidth)});
    dX.fill(0);

    auto& pool = GlobalThreadPool::getInstance();
    std::vector<std::future<void>> futures;
    futures.reserve(dimCache_.batchSize);

    for (int n = 0; n < dimCache_.batchSize; ++n) {
      futures.push_back(pool.enqueue([this, &dX, n]() {
        const int colOffset = n * dimCache_.numColumns;
        mat_t cols = colWorkspace_.block(0, colOffset, patchSize_, dimCache_.numColumns);
        ConvUtil::col2im(dX, cols, n, dimCache_);
      }));
    }

    for (auto& future : futures) {
      future.get();
    }

    return dX;
  }

  void _tensorToWorkspace(const Tensor& tensor) {
    const int outH = dimCache_.outputHeight;
    const int outW = dimCache_.outputWidth;

    for (int n = 0; n < dimCache_.batchSize; ++n) {
      const int colOffset = n * dimCache_.numColumns;

      for (int oc = 0; oc < outChannels_; ++oc) {
        for (int oh = 0; oh < outH; ++oh) {
          for (int ow = 0; ow < outW; ++ow) {
            const int col = oh * outW + ow;
            outWorkspace_(oc, colOffset + col) = tensor.at(n, oc, oh, ow);
          }
        }
      }
    }
  }

  Tensor _workspaceToTensor() const {
    Tensor Y = _createOutputTensor();

    const int outH = dimCache_.outputHeight;
    const int outW = dimCache_.outputWidth;

    // workspace layout: [outChannels x (batchSize * numColumns)]
    // tensor layout: [batchSize x outChannels x outH x outW]

    for (int n = 0; n < dimCache_.batchSize; ++n) {
      const int colOffset = n * dimCache_.numColumns;

      for (int oc = 0; oc < outChannels_; ++oc) {
        for (int oh = 0; oh < outH; ++oh) {
          for (int ow = 0; ow < outW; ++ow) {
            const int col = oh * outW + ow;
            // workspace: (channel, batch*spatial) -> tensor: (batch, channel, h, w)
            Y.at(n, oc, oh, ow) = outWorkspace_(oc, colOffset + col);
          }
        }
      }
    }

    return Y;
  }

  void _initializeParameters(INIT random) {
    weights_ = mat_t::Zero(outChannels_, patchSize_);
    biases_ = vec_t::Zero(outChannels_);

    dW_accum_ = mat_t::Zero(outChannels_, patchSize_);
    dB_accum_ = vec_t::Zero(outChannels_);

    switch (random) {
      case INIT::XAVIER: {
        const int fanIn = inChannels_ * kernelHeight_ * kernelWidth_;
        const int fanOut = outChannels_ * kernelHeight_ * kernelWidth_;
        const val_t scale = std::sqrt(val_t(2) / static_cast<val_t>(fanIn + fanOut));
        _randomWeight(scale);
        break;
      }
      case INIT::HE: {
        const val_t scale = std::sqrt(val_t(2) / static_cast<val_t>(patchSize_));
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
  // Layer configuration
  int inChannels_;
  int outChannels_;
  int kernelHeight_;
  int kernelWidth_;
  int stride_;
  int pad_;
  int patchSize_;

  // Trainable parameters
  mat_t weights_;
  vec_t biases_;

  // Gradient accumulators
  mat_t dW_accum_;
  vec_t dB_accum_;
  size_t accumSteps_;

  // Workspace matrices
  mutable mat_t colWorkspace_;
  mutable mat_t outWorkspace_;

  // Cached for backward
  Tensor inputCache_;
  ConvDimensions dimCache_;
};
