#pragma once

#include "ThreadPool.h"
#include "base/BasePoolingLayer.h"

class AveragePoolingLayer : public BasePoolingLayer {
 public:
  // Constructor for training (auto-detect input dimensions)
  explicit AveragePoolingLayer(int channels, int poolWidth, int poolHeight, int stride = 1, int pad = 0)
      : BasePoolingLayer("AveragePooling", channels, poolWidth, poolHeight, stride, pad) {}

  // Constructor for deserialization (with known input dimensions)
  AveragePoolingLayer(WithInputDims, int inputWidth, int inputHeight, int channels, int poolWidth, int poolHeight,
      int stride = 1, int pad = 0)
      : BasePoolingLayer("AveragePooling", inputWidth, inputHeight, channels, poolWidth, poolHeight, stride, pad) {}

  virtual ~AveragePoolingLayer() = default;

 public:
  Tensor forward(const Tensor& input) override {
    if (input.dim() != 4) {
      throw LayerException(getName(), std::format("Expected 4D input, got {}D", input.dim()));
    }

    if (inputWidth_ == 0 || inputHeight_ == 0) {
      initDims(input);
    }

    const size_t batchSize = input.shape(0);
    Tensor output = createOutputTensor(batchSize);

    if (batchSize * channels_ >= 4) {
      _forwardParallel(input, output, batchSize);
    } else {
      _forward(input, output, batchSize);
    }

    lastInput_ = input;
    return output;
  }

  Tensor backward(const Tensor& dY) override {
    const size_t batchSize = dY.shape(0);
    Tensor dX = createInputGradientTensor(batchSize);

    if (batchSize * channels_ >= 4) {
      _backwardParallel(dY, dX, batchSize);
    } else {
      _backward(dY, dX, batchSize);
    }

    return dX;
  }

 private:
  void _forward(const Tensor& input, Tensor& output, size_t batchSize) {
    const val_t* inData = input.data();
    val_t* outData = output.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;
    const float invPatchSize = 1.0f / static_cast<float>(patchSize_);

    for (size_t n = 0; n < batchSize; ++n) {
      const val_t* inSample = inData + (n * channels_ * inChannelStride);
      val_t* outSample = outData + (n * channels_ * outChannelStride);

      for (int c = 0; c < channels_; ++c) {
        const val_t* inChannel = inSample + (c * inChannelStride);
        val_t* outChannel = outSample + (c * outChannelStride);

        _poolChannelAverage(inChannel, outChannel, invPatchSize);
      }
    }
  }

  void _forwardParallel(const Tensor& input, Tensor& output, size_t batchSize) {
    auto& pool = GlobalThreadPool::getInstance();
    const val_t* inData = input.data();
    val_t* outData = output.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;
    const float invPatchSize = 1.0f / static_cast<float>(patchSize_);

    std::vector<std::future<void>> futures;
    futures.reserve(batchSize * channels_);

    for (size_t n = 0; n < batchSize; ++n) {
      for (int c = 0; c < channels_; ++c) {
        futures.push_back(
            pool.enqueue([this, inData, outData, n, c, inChannelStride, outChannelStride, invPatchSize]() {
              const val_t* inSample = inData + (n * channels_ * inChannelStride);
              val_t* outSample = outData + (n * channels_ * outChannelStride);

              const val_t* inChannel = inSample + (c * inChannelStride);
              val_t* outChannel = outSample + (c * outChannelStride);

              _poolChannelAverage(inChannel, outChannel, invPatchSize);
            }));
      }
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  // Average pool a single channel
  void _poolChannelAverage(const val_t* inChannel, val_t* outChannel, float invPatchSize) const {
    int outIdx = 0;

    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow, ++outIdx) {
        float sum = 0.0f;
        int count = 0;

        const int hStart = oh * stride_ - pad_;
        const int wStart = ow * stride_ - pad_;
        const int hEnd = std::min(hStart + poolHeight_, inputHeight_);
        const int wEnd = std::min(wStart + poolWidth_, inputWidth_);

        const int hBegin = std::max(hStart, 0);
        const int wBegin = std::max(wStart, 0);

        for (int ih = hBegin; ih < hEnd; ++ih) {
          for (int iw = wBegin; iw < wEnd; ++iw) {
            const int inIdx = ih * inputWidth_ + iw;
            sum += inChannel[inIdx];
            ++count;
          }
        }

        outChannel[outIdx] = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
      }
    }
  }

  void _backward(const Tensor& dY, Tensor& dX, size_t batchSize) const {
    const val_t* dYdata = dY.data();
    val_t* dXdata = dX.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;

    for (size_t n = 0; n < batchSize; ++n) {
      const val_t* dYsample = dYdata + (n * channels_ * outChannelStride);
      val_t* dXsample = dXdata + (n * channels_ * inChannelStride);

      for (int c = 0; c < channels_; ++c) {
        const val_t* dYchannel = dYsample + (c * outChannelStride);
        val_t* dXchannel = dXsample + (c * inChannelStride);

        _distributeGradientsChannel(dYchannel, dXchannel);
      }
    }
  }

  void _backwardParallel(const Tensor& dY, Tensor& dX, size_t batchSize) const {
    auto& pool = GlobalThreadPool::getInstance();
    const val_t* dYdata = dY.data();
    val_t* dXdata = dX.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;

    std::vector<std::future<void>> futures;
    futures.reserve(batchSize * channels_);

    for (size_t n = 0; n < batchSize; ++n) {
      for (int c = 0; c < channels_; ++c) {
        futures.push_back(pool.enqueue([this, dYdata, dXdata, n, c, inChannelStride, outChannelStride]() {
          const val_t* dYsample = dYdata + (n * channels_ * outChannelStride);
          val_t* dXsample = dXdata + (n * channels_ * inChannelStride);

          const val_t* dYchannel = dYsample + (c * outChannelStride);
          val_t* dXchannel = dXsample + (c * inChannelStride);

          _distributeGradientsChannel(dYchannel, dXchannel);
        }));
      }
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  void _distributeGradientsChannel(const val_t* dYchannel, val_t* dXchannel) const {
    int outIdx = 0;

    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow, ++outIdx) {
        const float gradOut = dYchannel[outIdx];

        const int hStart = oh * stride_ - pad_;
        const int wStart = ow * stride_ - pad_;
        const int hEnd = std::min(hStart + poolHeight_, inputHeight_);
        const int wEnd = std::min(wStart + poolWidth_, inputWidth_);

        const int hBegin = std::max(hStart, 0);
        const int wBegin = std::max(wStart, 0);

        const int validH = hEnd - hBegin;
        const int validW = wEnd - wBegin;
        const int count = validH * validW;

        if (count == 0)
          continue;

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