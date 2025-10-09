#pragma once

#include "Network.h"

class NetworkBuilder {
 public:
  Network build() {
    network_.infos();
    return std::move(network_);
  }

  NetworkBuilder& relu() {
    network_.addLayer(std::make_unique<ReLULayer>());
    return *this;
  }

  NetworkBuilder& lrelu() {
    network_.addLayer(std::make_unique<LReLULayer>());
    return *this;
  }

  NetworkBuilder& sigmoid() {
    network_.addLayer(std::make_unique<SigmoidLayer>());
    return *this;
  }

  NetworkBuilder& softmax() {
    network_.addLayer(std::make_unique<SoftmaxLayer>());
    return *this;
  }

  NetworkBuilder& tanh() {
    network_.addLayer(std::make_unique<TanhLayer>());
    return *this;
  }

  NetworkBuilder& batchNorm(int numFeatures, double momentum = 0.9, double epsilon = 1e-5, bool isTraining = false) {
    network_.addLayer(std::make_unique<BatchNormLayer>(numFeatures, momentum, epsilon, isTraining));
    return *this;
  }

  NetworkBuilder& conv(int inChannels, int outChannels, int kernelHeight, int kernelWidth, int stride = 1, int pad = 0,
      INIT random = INIT::XAVIER) {
    network_.addLayer(
        std::make_unique<Conv2DLayer>(inChannels, outChannels, kernelHeight, kernelWidth, stride, pad, random));
    return *this;
  }

  NetworkBuilder& dropout(double rate = 0.5, bool isTraining = false) {
    network_.addLayer(std::make_unique<DropoutLayer>(rate, isTraining));
    return *this;
  }

  NetworkBuilder& flatten() {
    network_.addLayer(std::make_unique<FlattenLayer>());
    return *this;
  }

  NetworkBuilder& fc(int in, int out, INIT init = INIT::XAVIER) {
    network_.addLayer(std::make_unique<FullyConnectedLayer>(in, out, init));
    return *this;
  }

  // type : ["avg", "max"]
  NetworkBuilder& pool(const std::string& type, int channels, int poolWidth, int poolHeight, int stride = 1,
      int pad = 0) {
    if (type == "avg") {
      network_.addLayer(std::make_unique<AveragePoolingLayer>(channels, poolWidth, poolHeight, stride, pad));
    }
    if (type == "max") {
      network_.addLayer(std::make_unique<MaxPoolingLayer>(channels, poolWidth, poolHeight, stride, pad));
    }
    return *this;
  }

  // type : ["avg", "max"]
  NetworkBuilder& pool(const std::string& type, BasePoolingLayer::WithInputDims tag, int inputWidth, int inputHeight,
      int channels, int poolWidth, int poolHeight, int stride = 1, int pad = 0) {
    if (type == "avg") {
      network_.addLayer(std::make_unique<AveragePoolingLayer>(tag, inputWidth, inputHeight, channels, poolWidth,
          poolHeight, stride, pad));
    }
    if (type == "max") {
      network_.addLayer(std::make_unique<MaxPoolingLayer>(tag, inputWidth, inputHeight, channels, poolWidth, poolHeight,
          stride, pad));
    }
    return *this;
  }

 private:
  Network network_;
};