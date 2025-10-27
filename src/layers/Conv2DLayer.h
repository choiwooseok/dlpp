#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <string>
#include <vector>

#include "layers/base/BaseLayer.h"
#include "types.h"

class Conv2DLayer : public BaseLayer {
 public:
  Conv2DLayer(int inChannels,
              int outChannels,
              int kernelHeight,
              int kernelWidth,
              int stride = 1,
              int pad = 0,
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

  tensor_t forward(const tensor_t &X) override {
    assert(X.ndim() == 4);
    assert(static_cast<int>(X.shape[1]) == inChannels_);

    const auto dims = _extractDims(X);
    if (dims.batchSize == 0) {
      return _createOutputTensor(dims);
    }

    _ensureWorkspaceSize(dims);

    // Batched im2col transformation (in-place into workspace)
    _buildBatchedColumnsInPlace(X, dims);

    // Batched convolution via GEMM (in-place into output workspace)
    // Single GEMM: Y = W * X + b
    outWorkspace_.noalias() = weights_ * colWorkspace_;
    outWorkspace_.colwise() += biases_;

    // Convert workspace to output tensor
    tensor_t Y = _convertWorkspaceToTensor(dims);

    lastInput_ = X;
    return Y;
  }

  tensor_t backward(const tensor_t &dY) override {
    assert(lastInput_.ndim() == 4);

    const auto dims = _extractDims(lastInput_);
    if (dims.batchSize == 0) {
      return tensor_t(lastInput_.shape);
    }

    // Build batched matrices
    _ensureWorkspaceSize(dims);
    _buildBatchedColumnsInPlace(lastInput_, dims);
    _buildGradientMatrixInPlace(dY, dims);

    // Accumulate parameter gradients (GEMM)
    dW_accum_.noalias() += outWorkspace_ * colWorkspace_.transpose();
    dB_accum_.noalias() += outWorkspace_.rowwise().sum();
    accumSteps_ += dims.batchSize;

    // Compute input gradients (Reuse colWorkspace for gradient)
    colWorkspace_.noalias() = weights_.transpose() * outWorkspace_;

    // Scatter back to input space (col2im)
    tensor_t dX = _scatterToInput(dims);

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
    cout << "["
         << "inC=" << inChannels_
         << " outC=" << outChannels_
         << " k=" << kernelHeight_ << "x" << kernelWidth_
         << " s=" << stride_
         << " p=" << pad_
         << "]";
  }

 public:
  // Getters
  int getInChannels() const { return inChannels_; }
  int getOutChannels() const { return outChannels_; }
  int getKernelHeight() const { return kernelHeight_; }
  int getKernelWidth() const { return kernelWidth_; }
  int getStride() const { return stride_; }
  int getPadding() const { return pad_; }

  const mat_t &getWeights() const { return weights_; }
  void setWeights(const mat_t &weights) { weights_ = weights; }

  const vec_t &getBiases() const { return biases_; }
  void setBiases(const vec_t &biases) { biases_ = biases; }

 private:
  struct ConvDimensions {
    int batchSize;
    int inputChannels;
    int inputHeight;
    int inputWidth;
    int outputHeight;
    int outputWidth;
    int numColumns;
  };

  ConvDimensions _extractDims(const tensor_t &X) const {
    ConvDimensions dims;
    dims.batchSize = static_cast<int>(X.shape[0]);
    dims.inputChannels = static_cast<int>(X.shape[1]);
    dims.inputHeight = static_cast<int>(X.shape[2]);
    dims.inputWidth = static_cast<int>(X.shape[3]);
    dims.outputHeight = (dims.inputHeight + 2 * pad_ - kernelHeight_) / stride_ + 1;
    dims.outputWidth = (dims.inputWidth + 2 * pad_ - kernelWidth_) / stride_ + 1;
    dims.numColumns = dims.outputHeight * dims.outputWidth;
    return dims;
  }

  tensor_t _createOutputTensor(const ConvDimensions &dims) const {
    return tensor_t({static_cast<size_t>(dims.batchSize),
                     static_cast<size_t>(outChannels_),
                     static_cast<size_t>(dims.outputHeight),
                     static_cast<size_t>(dims.outputWidth)});
  }

  // Workspace management for memory reuse
  void _ensureWorkspaceSize(const ConvDimensions &dims) {
    const int totalCols = dims.batchSize * dims.numColumns;

    // Resize workspaces only if needed
    if (colWorkspace_.rows() != patchSize_ || colWorkspace_.cols() != totalCols) {
      colWorkspace_.resize(patchSize_, totalCols);
    }
    if (outWorkspace_.rows() != outChannels_ || outWorkspace_.cols() != totalCols) {
      outWorkspace_.resize(outChannels_, totalCols);
    }
  }

  void _buildBatchedColumnsInPlace(const tensor_t &X, const ConvDimensions &dims) {
    // Parallel im2col for each sample
    for (int n = 0; n < dims.batchSize; ++n) {
      const int colOffset = n * dims.numColumns;

      // Direct im2col into workspace block
      auto K = X.im2colSample(n, kernelHeight_, kernelWidth_, stride_, pad_);
      colWorkspace_.block(0, colOffset, patchSize_, dims.numColumns) = K;
    }
  }

  tensor_t _convertWorkspaceToTensor(const ConvDimensions &dims) const {
    tensor_t Y = _createOutputTensor(dims);
    const int outW = dims.outputWidth;

    // Optimized copy with direct pointer access
    val_t *yData = Y.data.data();
    const val_t *workData = outWorkspace_.data();

    for (int n = 0; n < dims.batchSize; ++n) {
      const int colOffset = n * dims.numColumns;
      val_t *ySample = yData + (n * outChannels_ * dims.numColumns);

      for (int oc = 0; oc < outChannels_; ++oc) {
        const val_t *workRow = workData + (oc * outWorkspace_.cols() + colOffset);
        val_t *yChannel = ySample + (oc * dims.numColumns);

        // Memcpy for contiguous data
        std::memcpy(yChannel, workRow, dims.numColumns * sizeof(val_t));
      }
    }

    return Y;
  }

  void _buildGradientMatrixInPlace(const tensor_t &dY, const ConvDimensions &dims) {
    const int outW = dims.outputWidth;
    const int outH = dims.outputHeight;

    // Direct copy into workspace
    for (int n = 0; n < dims.batchSize; ++n) {
      const int colOffset = n * dims.numColumns;

      for (int oc = 0; oc < outChannels_; ++oc) {
        for (int oh = 0; oh < outH; ++oh) {
          for (int ow = 0; ow < outW; ++ow) {
            const int col = oh * outW + ow;
            outWorkspace_(oc, colOffset + col) = dY.at(n, oc, oh, ow);
          }
        }
      }
    }
  }

  tensor_t _scatterToInput(const ConvDimensions &dims) const {
    tensor_t dX({static_cast<size_t>(dims.batchSize),
                 static_cast<size_t>(dims.inputChannels),
                 static_cast<size_t>(dims.inputHeight),
                 static_cast<size_t>(dims.inputWidth)});
    dX.fill(0);

    // Optimized col2im with cache-friendly access
    for (int n = 0; n < dims.batchSize; ++n) {
      const int colOffset = n * dims.numColumns;

      for (int col = 0; col < dims.numColumns; ++col) {
        const int oh = col / dims.outputWidth;
        const int ow = col % dims.outputWidth;

        int patchIdx = 0;
        for (int ic = 0; ic < inChannels_; ++ic) {
          for (int kh = 0; kh < kernelHeight_; ++kh) {
            for (int kw = 0; kw < kernelWidth_; ++kw) {
              const int ih = oh * stride_ + kh - pad_;
              const int iw = ow * stride_ + kw - pad_;

              if (ih >= 0 && ih < dims.inputHeight && iw >= 0 && iw < dims.inputWidth) {
                dX.at(n, ic, ih, iw) += colWorkspace_(patchIdx, colOffset + col);
              }
              ++patchIdx;
            }
          }
        }
      }
    }

    return dX;
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
    for (int r = 0; r < weights_.rows(); ++r) {
      for (int c = 0; c < weights_.cols(); ++c) {
        weights_(r, c) = genRandom() * scale;
      }
    }
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

  // Workspace matrices (reused across forward/backward)
  mutable mat_t colWorkspace_;  // im2col result
  mutable mat_t outWorkspace_;  // convolution output

  // Cached for backward pass
  tensor_t lastInput_;
};