#pragma once

#include "base/BasePoolingLayer.h"

class MaxPoolingLayer : public BasePoolingLayer {
 public:
  // Constructor for training (auto-detect input dimensions)
  explicit MaxPoolingLayer(int channels,
                           int poolWidth,
                           int poolHeight,
                           int stride = 1,
                           int pad = 0)
      : BasePoolingLayer("MaxPooling",
                         channels,
                         poolWidth,
                         poolHeight,
                         stride,
                         pad) {
  }

  // Constructor for deserialization (with known input dimensions)
  MaxPoolingLayer(WithInputDims,
                  int inputWidth,
                  int inputHeight,
                  int channels,
                  int poolWidth,
                  int poolHeight,
                  int stride = 1,
                  int pad = 0)
      : BasePoolingLayer("MaxPooling",
                         inputWidth,
                         inputHeight,
                         channels,
                         poolWidth,
                         poolHeight,
                         stride,
                         pad) {
  }

  virtual ~MaxPoolingLayer() = default;

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

    // Prepare output and indices storage
    tensor_t output = createOutputTensor(batchSize);
    _ensureIndicesCapacity(batchSize);

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
  void _ensureIndicesCapacity(size_t batchSize) {
    const size_t requiredSize = batchSize * channels_ * outputHeight_ * outputWidth_;
    if (maxIndices_.size() != requiredSize) {
      maxIndices_.resize(requiredSize);
    }
  }

  // forward pass with direct pointer access
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
        _poolChannel(inChannel, outChannel, indicesChannel);
      }
    }
  }

  // Pool a single channel with optimized loops
  void _poolChannel(const val_t *inChannel, val_t *outChannel, int *indices) const {
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

  // backward pass with direct pointer access
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
      const int *indicesSample = indicesData + (n * channels_ * outChannelStride);

      // Process each channel
      for (int c = 0; c < channels_; ++c) {
        const val_t *dYchannel = dYsample + (c * outChannelStride);
        val_t *dXchannel = dXsample + (c * inChannelStride);
        const int *indicesChannel = indicesSample + (c * outChannelStride);

        // Route gradients to max positions
        _routeGradientsChannel(dYchannel, dXchannel, indicesChannel);
      }
    }
  }

  // Route gradients for a single channel
  void _routeGradientsChannel(const val_t *dYchannel, val_t *dXchannel, const int *indices) const {
    const int numOutputs = outputHeight_ * outputWidth_;

    for (int i = 0; i < numOutputs; ++i) {
      const int maxIdx = indices[i];
      if (maxIdx >= 0) {  // Valid index
        dXchannel[maxIdx] += dYchannel[i];
      }
    }
  }

 private:
  std::vector<int> maxIndices_;  // Flat indices for max positions
};