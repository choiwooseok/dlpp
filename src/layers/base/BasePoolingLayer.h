#pragma once

#include "BaseLayer.h"

class BasePoolingLayer : public BaseLayer {
 public:
  // Tag for constructor disambiguation
  struct WithInputDims {};
  static constexpr WithInputDims with_input_dims{};

  // Constructor for training (auto-detect input dimensions)
  explicit BasePoolingLayer(const std::string& name, int channels, int poolWidth, int poolHeight, int stride, int pad)
      : BaseLayer(name),
        inputWidth_(0),
        inputHeight_(0),
        channels_(channels),
        poolWidth_(poolWidth),
        poolHeight_(poolHeight),
        stride_(stride),
        pad_(pad),
        outputWidth_(0),
        outputHeight_(0),
        patchSize_(poolHeight * poolWidth) {}

  // Constructor for deserialization (with known input dimensions)
  BasePoolingLayer(const std::string& name, int inputWidth, int inputHeight, int channels, int poolWidth,
      int poolHeight, int stride, int pad)
      : BaseLayer(name),
        inputWidth_(inputWidth),
        inputHeight_(inputHeight),
        channels_(channels),
        poolWidth_(poolWidth),
        poolHeight_(poolHeight),
        stride_(stride),
        pad_(pad),
        patchSize_(poolHeight * poolWidth) {
    outputWidth_ = _calcDim(inputWidth, poolWidth, stride, pad);
    outputHeight_ = _calcDim(inputHeight, poolHeight, stride, pad);

    std::cout << std::format("{} initialized with input dims: {}x{}, output dims: {}x{}", getName(), inputHeight_,
                     inputWidth_, outputHeight_, outputWidth_)
              << std::endl;
  }

  virtual ~BasePoolingLayer() = default;

 public:
  // Pooling has no trainable parameters
  void updateParams(Optimizer* /*optimizer*/) override {}

  void info() override {
    bool autoDetect = (inputWidth_ == 0 && inputHeight_ == 0 && outputWidth_ == 0 && outputHeight_ == 0);

    if (!autoDetect) {
      std::cout << std::format("[in={}x{} ch={} pool={}x{} s={} p={} out={}x{}]", inputHeight_, inputWidth_, channels_,
          poolHeight_, poolWidth_, stride_, pad_, outputHeight_, outputWidth_);
    } else {
      std::cout << std::format("[in=auto ch={} pool={}x{} s={} p={} out=auto]", channels_, poolHeight_, poolWidth_,
          stride_, pad_);
    }
  }

 public:
  // Getters
  int getInputWidth() const {
    return inputWidth_;
  }
  int getInputHeight() const {
    return inputHeight_;
  }
  int getChannels() const {
    return channels_;
  }
  int getPoolWidth() const {
    return poolWidth_;
  }
  int getPoolHeight() const {
    return poolHeight_;
  }
  int getStride() const {
    return stride_;
  }
  int getPadding() const {
    return pad_;
  }
  int getOutputWidth() const {
    return outputWidth_;
  }
  int getOutputHeight() const {
    return outputHeight_;
  }

 private:
  int _calcDim(int inputDim, int poolDim, int stride, int pad) const {
    return (inputDim + 2 * pad - poolDim) / stride + 1;
  }

 protected:
  void initDims(const Tensor& input) {
    inputWidth_ = static_cast<int>(input.shape(3));
    inputHeight_ = static_cast<int>(input.shape(2));
    outputWidth_ = _calcDim(inputWidth_, poolWidth_, stride_, pad_);
    outputHeight_ = _calcDim(inputHeight_, poolHeight_, stride_, pad_);

    std::cout << std::format("{} initialized with input dims: {}x{}, output dims: {}x{}", getName(), inputHeight_,
                     inputWidth_, outputHeight_, outputWidth_)
              << std::endl;
  }

  Tensor createOutputTensor(size_t batchSize) const {
    return Tensor({batchSize, static_cast<size_t>(channels_), static_cast<size_t>(outputHeight_),
        static_cast<size_t>(outputWidth_)});
  }

  Tensor createInputGradientTensor(size_t batchSize) const {
    Tensor dX({batchSize, static_cast<size_t>(channels_), static_cast<size_t>(inputHeight_),
        static_cast<size_t>(inputWidth_)});
    dX.fill(0);
    return dX;
  }

 protected:
  // Input dimensions
  int inputWidth_;
  int inputHeight_;
  int channels_;

  // Pooling parameters
  int poolWidth_;
  int poolHeight_;
  int stride_;
  int pad_;

  // Output dimensions
  int outputWidth_;
  int outputHeight_;

  // Cached values
  int patchSize_;

  // Stored for backward pass
  Tensor lastInput_;
};