#pragma once

#include "base/BasePoolingLayer.h"

class AveragePoolingLayer : public BasePoolingLayer {
 public:
  // Constructor for training (auto-detect input dimensions)
  explicit AveragePoolingLayer(int channels,
                               int poolWidth,
                               int poolHeight,
                               int stride = 1,
                               int pad = 0)
      : BasePoolingLayer("AveragePooling",
                         channels,
                         poolWidth,
                         poolHeight,
                         stride,
                         pad) {
  }

  // Constructor for deserialization (with known input dimensions)
  AveragePoolingLayer(WithInputDims,
                      int inputWidth,
                      int inputHeight,
                      int channels,
                      int poolWidth,
                      int poolHeight,
                      int stride = 1,
                      int pad = 0)
      : BasePoolingLayer("AveragePooling",
                         inputWidth,
                         inputHeight,
                         channels,
                         poolWidth,
                         poolHeight,
                         stride,
                         pad) {
  }

  virtual ~AveragePoolingLayer() = default;

 public:
  tensor_t forward(const tensor_t &input) override {
    assert(input.ndim() == 4);

    // Initialize dimensions on first forward pass
    if (inputWidth_ == 0 || inputHeight_ == 0) {
      initDims(input);
    }

    // Validate input
    assert(static_cast<int>(input.shape[1]) == channels_);
    assert(static_cast<int>(input.shape[2]) == inputHeight_);
    assert(static_cast<int>(input.shape[3]) == inputWidth_);

    const size_t batchSize = input.shape[0];
    if (batchSize == 0) {
      return createOutputTensor(0);
    }

    tensor_t output = createOutputTensor(batchSize);

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

 private:
  // forward pass with direct pointer access
  void _forward(const tensor_t &input, tensor_t &output, size_t batchSize) {
    const val_t *inData = input.data.data();
    val_t *outData = output.data.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;
    const float invPatchSize = 1.0f / static_cast<float>(patchSize_);

    // Process each sample
    for (size_t n = 0; n < batchSize; ++n) {
      const val_t *inSample = inData + (n * channels_ * inChannelStride);
      val_t *outSample = outData + (n * channels_ * outChannelStride);

      // Process each channel
      for (int c = 0; c < channels_; ++c) {
        const val_t *inChannel = inSample + (c * inChannelStride);
        val_t *outChannel = outSample + (c * outChannelStride);

        // Process spatial dimensions
        _poolChannelAverage(inChannel, outChannel, invPatchSize);
      }
    }
  }

  // Average pool a single channel with optimized loops
  void _poolChannelAverage(const val_t *inChannel, val_t *outChannel, float invPatchSize) const {
    int outIdx = 0;

    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow, ++outIdx) {
        float sum = 0.0f;
        int count = 0;

        // Compute pooling window bounds
        const int hStart = oh * stride_ - pad_;
        const int wStart = ow * stride_ - pad_;
        const int hEnd = std::min(hStart + poolHeight_, inputHeight_);
        const int wEnd = std::min(wStart + poolWidth_, inputWidth_);

        const int hBegin = std::max(hStart, 0);
        const int wBegin = std::max(wStart, 0);

        // Sum values in pooling window
        for (int ih = hBegin; ih < hEnd; ++ih) {
          for (int iw = wBegin; iw < wEnd; ++iw) {
            const int inIdx = ih * inputWidth_ + iw;
            sum += inChannel[inIdx];
            ++count;
          }
        }

        // Compute average (handle padding by using actual count)
        outChannel[outIdx] = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
      }
    }
  }

  // backward pass with direct pointer access
  void _backward(const tensor_t &dY, tensor_t &dX, size_t batchSize) const {
    const val_t *dYdata = dY.data.data();
    val_t *dXdata = dX.data.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;

    // Process each sample
    for (size_t n = 0; n < batchSize; ++n) {
      const val_t *dYsample = dYdata + (n * channels_ * outChannelStride);
      val_t *dXsample = dXdata + (n * channels_ * inChannelStride);

      // Process each channel
      for (int c = 0; c < channels_; ++c) {
        const val_t *dYchannel = dYsample + (c * outChannelStride);
        val_t *dXchannel = dXsample + (c * inChannelStride);

        // Distribute gradients
        _distributeGradientsChannel(dYchannel, dXchannel);
      }
    }
  }

  // Distribute gradients for a single channel
  void _distributeGradientsChannel(const val_t *dYchannel, val_t *dXchannel) const {
    int outIdx = 0;

    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow, ++outIdx) {
        const float gradOut = dYchannel[outIdx];

        // Compute pooling window bounds
        const int hStart = oh * stride_ - pad_;
        const int wStart = ow * stride_ - pad_;
        const int hEnd = std::min(hStart + poolHeight_, inputHeight_);
        const int wEnd = std::min(wStart + poolWidth_, inputWidth_);

        const int hBegin = std::max(hStart, 0);
        const int wBegin = std::max(wStart, 0);

        // Count valid elements in window
        const int validH = hEnd - hBegin;
        const int validW = wEnd - wBegin;
        const int count = validH * validW;

        if (count == 0)
          continue;

        // Distribute gradient equally to all positions in window
        const float gradDistributed = gradOut / static_cast<float>(count);

        for (int ih = hBegin; ih < hEnd; ++ih) {
          for (int iw = wBegin; iw < wEnd; ++iw) {
            const int inIdx = ih * inputWidth_ + iw;
            dXchannel[inIdx] += gradDistributed;
          }
        }
      }
    }
  }
};