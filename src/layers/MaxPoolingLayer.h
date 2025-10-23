#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

#include "base/BaseLayer.h"

class MaxPoolingLayer : public BaseLayer {
public:
  // Tag for constructor disambiguation
  struct WithInputDims {};
  static constexpr WithInputDims with_input_dims{};

  // Constructor for training (auto-detect input dimensions)
  explicit MaxPoolingLayer(int channels, int poolWidth, int poolHeight,
                           int stride = 1, int pad = 0)
      : BaseLayer("MaxPooling"), inputWidth_(0), inputHeight_(0),
        channels_(channels), poolWidth_(poolWidth), poolHeight_(poolHeight),
        stride_(stride), pad_(pad), outputWidth_(0), outputHeight_(0),
        patchSize_(poolHeight * poolWidth) {}

  // Constructor for deserialization (with known input dimensions)
  MaxPoolingLayer(WithInputDims, int inputWidth, int inputHeight, int channels,
                  int poolWidth, int poolHeight, int stride = 1, int pad = 0)
      : BaseLayer("MaxPooling"), inputWidth_(inputWidth),
        inputHeight_(inputHeight), channels_(channels), poolWidth_(poolWidth),
        poolHeight_(poolHeight), stride_(stride), pad_(pad),
        patchSize_(poolHeight * poolWidth) {
    outputWidth_ = calculateOutputDim(inputWidth, poolWidth, stride, pad);
    outputHeight_ = calculateOutputDim(inputHeight, poolHeight, stride, pad);
  }

  virtual ~MaxPoolingLayer() = default;

  tensor_t forward(const tensor_t &input) override {
    assert(input.ndim() == 4);

    // Initialize dimensions on first forward pass
    if (inputWidth_ == 0 || inputHeight_ == 0) {
      initializeDimensions(input);
    }

    // Validate input
    assert(static_cast<int>(input.shape[1]) == channels_);
    assert(static_cast<int>(input.shape[2]) == inputHeight_);
    assert(static_cast<int>(input.shape[3]) == inputWidth_);

    const size_t batchSize = input.shape[0];
    if (batchSize == 0) {
      return createOutputTensor(0);
    }

    // Prepare output and indices storage
    tensor_t output = createOutputTensor(batchSize);
    ensureIndicesCapacity(batchSize);

    // forward pass
    _forward(input, output, batchSize);

    lastInput_ = input;
    return output;
  }

  tensor_t backward(const tensor_t &dY) override {
    assert(dY.ndim() == 4);
    assert(static_cast<int>(dY.shape[1]) == channels_);
    assert(static_cast<int>(dY.shape[2]) == outputHeight_);
    assert(static_cast<int>(dY.shape[3]) == outputWidth_);

    const size_t batchSize = dY.shape[0];
    if (batchSize == 0) {
      return createInputGradientTensor(0);
    }

    tensor_t dX = createInputGradientTensor(batchSize);

    // backward pass
    _backward(dY, dX, batchSize);

    return dX;
  }

  void updateParams(double /*eta*/) override {
    // MaxPooling has no trainable parameters
  }

  void info() override {
    cout << "[in=" << inputWidth_ << "x" << inputHeight_ << " ch=" << channels_
         << " pool=" << poolWidth_ << "x" << poolHeight_ << " s=" << stride_
         << " p=" << pad_ << " out=" << outputWidth_ << "x" << outputHeight_
         << "]";
  }

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
  static int calculateOutputDim(int inputDim, int poolDim, int stride,
                                int pad) {
    return (inputDim + 2 * pad - poolDim) / stride + 1;
  }

  void initializeDimensions(const tensor_t &input) {
    inputWidth_ = static_cast<int>(input.shape[3]);
    inputHeight_ = static_cast<int>(input.shape[2]);
    outputWidth_ = calculateOutputDim(inputWidth_, poolWidth_, stride_, pad_);
    outputHeight_ =
        calculateOutputDim(inputHeight_, poolHeight_, stride_, pad_);
  }

  tensor_t createOutputTensor(size_t batchSize) const {
    return tensor_t({batchSize, static_cast<size_t>(channels_),
                     static_cast<size_t>(outputHeight_),
                     static_cast<size_t>(outputWidth_)});
  }

  tensor_t createInputGradientTensor(size_t batchSize) const {
    tensor_t dX({batchSize, static_cast<size_t>(channels_),
                 static_cast<size_t>(inputHeight_),
                 static_cast<size_t>(inputWidth_)});
    dX.fill(0);
    return dX;
  }

  void ensureIndicesCapacity(size_t batchSize) {
    const size_t requiredSize =
        batchSize * channels_ * outputHeight_ * outputWidth_;
    if (maxIndices_.size() != requiredSize) {
      maxIndices_.resize(requiredSize);
    }
  }

  // Optimized forward pass with better memory access pattern
  void _forward(const tensor_t &input, tensor_t &output, size_t batchSize) {
    // Direct pointer access for better performance
    const val_t *inData = input.data.data();
    val_t *outData = output.data.data();
    int *indicesData = maxIndices_.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;

    // Process each sample
    for (size_t n = 0; n < batchSize; ++n) {
      const val_t *inSample = inData + (n * channels_ * inChannelStride);
      val_t *outSample = outData + (n * channels_ * outChannelStride);
      int *indicesSample = indicesData + (n * channels_ * outChannelStride);

      // Process each channel
      for (int c = 0; c < channels_; ++c) {
        const val_t *inChannel = inSample + (c * inChannelStride);
        val_t *outChannel = outSample + (c * outChannelStride);
        int *indicesChannel = indicesSample + (c * outChannelStride);

        // Process spatial dimensions
        poolChannel(inChannel, outChannel, indicesChannel);
      }
    }
  }

  // Pool a single channel with optimized loops
  void poolChannel(const val_t *inChannel, val_t *outChannel,
                   int *indices) const {
    int outIdx = 0;

    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow, ++outIdx) {
        float maxVal = -std::numeric_limits<float>::infinity();
        int maxIdx = -1;

        // Find max in pooling window
        const int hStart = oh * stride_ - pad_;
        const int wStart = ow * stride_ - pad_;
        const int hEnd = std::min(hStart + poolHeight_, inputHeight_);
        const int wEnd = std::min(wStart + poolWidth_, inputWidth_);

        // Ensure we stay within bounds
        const int hBegin = std::max(hStart, 0);
        const int wBegin = std::max(wStart, 0);

        for (int ih = hBegin; ih < hEnd; ++ih) {
          for (int iw = wBegin; iw < wEnd; ++iw) {
            const int inIdx = ih * inputWidth_ + iw;
            const float val = inChannel[inIdx];

            if (val > maxVal) {
              maxVal = val;
              maxIdx = inIdx;
            }
          }
        }

        outChannel[outIdx] = maxVal;
        indices[outIdx] = maxIdx;
      }
    }
  }

  // Optimized backward pass with better memory access pattern
  void _backward(const tensor_t &dY, tensor_t &dX, size_t batchSize) const {
    const val_t *dYdata = dY.data.data();
    val_t *dXdata = dX.data.data();
    const int *indicesData = maxIndices_.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;

    // Process each sample
    for (size_t n = 0; n < batchSize; ++n) {
      const val_t *dYsample = dYdata + (n * channels_ * outChannelStride);
      val_t *dXsample = dXdata + (n * channels_ * inChannelStride);
      const int *indicesSample =
          indicesData + (n * channels_ * outChannelStride);

      // Process each channel
      for (int c = 0; c < channels_; ++c) {
        const val_t *dYchannel = dYsample + (c * outChannelStride);
        val_t *dXchannel = dXsample + (c * inChannelStride);
        const int *indicesChannel = indicesSample + (c * outChannelStride);

        // Route gradients to max positions
        routeGradientsChannel(dYchannel, dXchannel, indicesChannel);
      }
    }
  }

  // Route gradients for a single channel
  void routeGradientsChannel(const val_t *dYchannel, val_t *dXchannel,
                             const int *indices) const {
    const int numOutputs = outputHeight_ * outputWidth_;

    for (int i = 0; i < numOutputs; ++i) {
      const int maxIdx = indices[i];
      if (maxIdx >= 0) { // Valid index
        dXchannel[maxIdx] += dYchannel[i];
      }
    }
  }

private:
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
  std::vector<int> maxIndices_; // Flat indices for max positions
};