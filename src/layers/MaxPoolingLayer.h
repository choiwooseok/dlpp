#pragma once

#include "ThreadPool.h"
#include "base/BasePoolingLayer.h"

class MaxPoolingLayer : public BasePoolingLayer {
 public:
  // Constructor for training (auto-detect input dimensions)
  explicit MaxPoolingLayer(int channels, int poolWidth, int poolHeight, int stride = 1, int pad = 0)
      : BasePoolingLayer("MaxPooling", channels, poolWidth, poolHeight, stride, pad) {}

  // Constructor for deserialization (with known input dimensions)
  MaxPoolingLayer(WithInputDims, int inputWidth, int inputHeight, int channels, int poolWidth, int poolHeight,
      int stride = 1, int pad = 0)
      : BasePoolingLayer("MaxPooling", inputWidth, inputHeight, channels, poolWidth, poolHeight, stride, pad) {}

  virtual ~MaxPoolingLayer() = default;

  Tensor forward(const Tensor& input) override {
    if (input.dim() != 4) {
      throw LayerException(getName(), std::format("Expected 4D input, got {}D", input.dim()));
    }

    // Initialize dimensions on first forward pass
    if (inputWidth_ == 0 || inputHeight_ == 0) {
      initDims(input);
    }

    const size_t batchSize = input.shape(0);
    Tensor output = createOutputTensor(batchSize);
    _ensureIndicesCapacity(batchSize);

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
  void _ensureIndicesCapacity(size_t batchSize) {
    const size_t requiredSize = batchSize * channels_ * outputHeight_ * outputWidth_;
    if (maxIndices_.size() != requiredSize) {
      maxIndices_.resize(requiredSize);
    }
  }

  void _forward(const Tensor& input, Tensor& output, size_t batchSize) {
    const val_t* inData = input.data();
    val_t* outData = output.data();
    int* indicesData = maxIndices_.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;

    for (size_t n = 0; n < batchSize; ++n) {
      const val_t* inSample = inData + (n * channels_ * inChannelStride);
      val_t* outSample = outData + (n * channels_ * outChannelStride);
      int* indicesSample = indicesData + (n * channels_ * outChannelStride);

      for (int c = 0; c < channels_; ++c) {
        const val_t* inChannel = inSample + (c * inChannelStride);
        val_t* outChannel = outSample + (c * outChannelStride);
        int* indicesChannel = indicesSample + (c * outChannelStride);

        _poolChannel(inChannel, outChannel, indicesChannel);
      }
    }
  }

  void _forwardParallel(const Tensor& input, Tensor& output, size_t batchSize) {
    auto& pool = GlobalThreadPool::getInstance();
    const val_t* inData = input.data();
    val_t* outData = output.data();
    int* indicesData = maxIndices_.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;
    const size_t totalTasks = batchSize * channels_;

    std::vector<std::future<void>> futures;
    futures.reserve(totalTasks);

    for (size_t n = 0; n < batchSize; ++n) {
      for (int c = 0; c < channels_; ++c) {
        futures.push_back(pool.enqueue([this, inData, outData, indicesData, n, c, inChannelStride, outChannelStride]() {
          const val_t* inSample = inData + (n * channels_ * inChannelStride);
          val_t* outSample = outData + (n * channels_ * outChannelStride);
          int* indicesSample = indicesData + (n * channels_ * outChannelStride);

          const val_t* inChannel = inSample + (c * inChannelStride);
          val_t* outChannel = outSample + (c * outChannelStride);
          int* indicesChannel = indicesSample + (c * outChannelStride);

          _poolChannel(inChannel, outChannel, indicesChannel);
        }));
      }
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  void _poolChannel(const val_t* inChannel, val_t* outChannel, int* indices) const {
    int outIdx = 0;

    for (int oh = 0; oh < outputHeight_; ++oh) {
      for (int ow = 0; ow < outputWidth_; ++ow, ++outIdx) {
        float maxVal = -std::numeric_limits<float>::infinity();
        int maxIdx = -1;

        const int hStart = oh * stride_ - pad_;
        const int wStart = ow * stride_ - pad_;
        const int hEnd = std::min(hStart + poolHeight_, inputHeight_);
        const int wEnd = std::min(wStart + poolWidth_, inputWidth_);

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

  void _backward(const Tensor& dY, Tensor& dX, size_t batchSize) const {
    const val_t* dYdata = dY.data();
    val_t* dXdata = dX.data();
    const int* indicesData = maxIndices_.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;

    for (size_t n = 0; n < batchSize; ++n) {
      const val_t* dYsample = dYdata + (n * channels_ * outChannelStride);
      val_t* dXsample = dXdata + (n * channels_ * inChannelStride);
      const int* indicesSample = indicesData + (n * channels_ * outChannelStride);

      for (int c = 0; c < channels_; ++c) {
        const val_t* dYchannel = dYsample + (c * outChannelStride);
        val_t* dXchannel = dXsample + (c * inChannelStride);
        const int* indicesChannel = indicesSample + (c * outChannelStride);

        _routeGradientsChannel(dYchannel, dXchannel, indicesChannel);
      }
    }
  }

  void _backwardParallel(const Tensor& dY, Tensor& dX, size_t batchSize) const {
    auto& pool = GlobalThreadPool::getInstance();
    const val_t* dYdata = dY.data();
    val_t* dXdata = dX.data();
    const int* indicesData = maxIndices_.data();

    const int inChannelStride = inputHeight_ * inputWidth_;
    const int outChannelStride = outputHeight_ * outputWidth_;
    const size_t totalTasks = batchSize * channels_;

    std::vector<std::future<void>> futures;
    futures.reserve(totalTasks);

    for (size_t n = 0; n < batchSize; ++n) {
      for (int c = 0; c < channels_; ++c) {
        futures.push_back(pool.enqueue([this, dYdata, dXdata, indicesData, n, c, inChannelStride, outChannelStride]() {
          const val_t* dYsample = dYdata + (n * channels_ * outChannelStride);
          val_t* dXsample = dXdata + (n * channels_ * inChannelStride);
          const int* indicesSample = indicesData + (n * channels_ * outChannelStride);

          const val_t* dYchannel = dYsample + (c * outChannelStride);
          val_t* dXchannel = dXsample + (c * inChannelStride);
          const int* indicesChannel = indicesSample + (c * outChannelStride);

          _routeGradientsChannel(dYchannel, dXchannel, indicesChannel);
        }));
      }
    }

    for (auto& future : futures) {
      future.get();
    }
  }

  // Route gradients for a single channel
  void _routeGradientsChannel(const val_t* dYchannel, val_t* dXchannel, const int* indices) const {
    const int numOutputs = outputHeight_ * outputWidth_;

    for (int i = 0; i < numOutputs; ++i) {
      const int maxIdx = indices[i];
      if (maxIdx >= 0) {
        dXchannel[maxIdx] += dYchannel[i];
      }
    }
  }

 private:
  std::vector<int> maxIndices_;
};