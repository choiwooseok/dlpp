#pragma once

#include "BaseLayer.h"

class BasePoolingLayer : public BaseLayer {
 public:
  // Tag for constructor disambiguation
  struct WithInputDims {};
  static constexpr WithInputDims with_input_dims{};

  // Constructor for training (auto-detect input dimensions)
  explicit BasePoolingLayer(const string &name,
                            int channels,
                            int poolWidth,
                            int poolHeight,
                            int stride,
                            int pad)
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
        patchSize_(poolHeight * poolWidth) {
  }

  // Constructor for deserialization (with known input dimensions)
  BasePoolingLayer(const string &name,
                   int inputWidth,
                   int inputHeight,
                   int channels,
                   int poolWidth,
                   int poolHeight,
                   int stride,
                   int pad)
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

    cout << getName()
         << " initialized with input dims: "
         << inputWidth_ << "x" << inputHeight_
         << ", output dims: "
         << outputWidth_ << "x" << outputHeight_
         << endl;
  }

  virtual ~BasePoolingLayer() = default;

 public:
  // Pooling has no trainable parameters
  void updateParams(double /*eta*/) override {}

  void info() override {
    cout << "["
         << "in=" << inputWidth_ << "x" << inputHeight_
         << " ch=" << channels_
         << " pool=" << poolWidth_ << "x" << poolHeight_
         << " s=" << stride_
         << " p=" << pad_
         << " out=" << outputWidth_ << "x" << outputHeight_
         << "]";
  }

 public:
  // Getters
  int getInputWidth() const { return inputWidth_; }
  int getInputHeight() const { return inputHeight_; }
  int getChannels() const { return channels_; }
  int getPoolWidth() const { return poolWidth_; }
  int getPoolHeight() const { return poolHeight_; }
  int getStride() const { return stride_; }
  int getPadding() const { return pad_; }
  int getOutputWidth() const { return outputWidth_; }
  int getOutputHeight() const { return outputHeight_; }

 private:
  int _calcDim(int inputDim, int poolDim, int stride, int pad) const {
    return (inputDim + 2 * pad - poolDim) / stride + 1;
  }

 protected:
  void initDims(const tensor_t &input) {
    inputWidth_ = static_cast<int>(input.shape[3]);
    inputHeight_ = static_cast<int>(input.shape[2]);
    outputWidth_ = _calcDim(inputWidth_, poolWidth_, stride_, pad_);
    outputHeight_ = _calcDim(inputHeight_, poolHeight_, stride_, pad_);

    cout << getName()
         << " initialized with input dims: "
         << inputWidth_ << "x" << inputHeight_
         << ", output dims: " << outputWidth_ << "x" << outputHeight_
         << endl;
  }

  tensor_t createOutputTensor(size_t batchSize) const {
    return tensor_t({batchSize,
                     static_cast<size_t>(channels_),
                     static_cast<size_t>(outputHeight_),
                     static_cast<size_t>(outputWidth_)});
  }

  tensor_t createInputGradientTensor(size_t batchSize) const {
    tensor_t dX({batchSize,
                 static_cast<size_t>(channels_),
                 static_cast<size_t>(inputHeight_),
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
  tensor_t lastInput_;
};