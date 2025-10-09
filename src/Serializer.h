#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "layers/AveragePoolingLayer.h"
#include "layers/BatchNormLayer.h"
#include "layers/Conv2DLayer.h"
#include "layers/DropoutLayer.h"
#include "layers/FlattenLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/MaxPoolingLayer.h"
#include "layers/activations/LReLULayer.h"
#include "layers/activations/ReLULayer.h"
#include "layers/activations/SigmoidLayer.h"
#include "layers/activations/SoftmaxLayer.h"
#include "layers/activations/TanhLayer.h"
#include "layers/base/BaseLayer.h"

using json = nlohmann::json;

class Serializer {
 public:
  static json marshal(const std::vector<std::shared_ptr<BaseLayer>>& layers) {
    json model;
    for (const auto& layer : layers) {
      json layer_;
      layer_["type"] = layer->getName();

      if (layer->getName() == "FullyConnected") {
        _fc(layer_, static_cast<FullyConnectedLayer*>(layer.get()));
      } else if (layer->getName() == "Conv2D") {
        _conv(layer_, static_cast<Conv2DLayer*>(layer.get()));
      } else if (layer->getName() == "AveragePooling") {
        _avg(layer_, static_cast<AveragePoolingLayer*>(layer.get()));
      } else if (layer->getName() == "MaxPooling") {
        _max(layer_, static_cast<MaxPoolingLayer*>(layer.get()));
      } else if (layer->getName() == "BatchNorm") {
        _batchnorm(layer_, static_cast<BatchNormLayer*>(layer.get()));
      } else if (layer->getName() == "Dropout") {
        _dropout(layer_, static_cast<DropoutLayer*>(layer.get()));
      }
      model["layers"].push_back(layer_);
    }
    return model;
  }

  static void unmarshal(std::vector<std::shared_ptr<BaseLayer>>& layers, const json& model) {
    for (const auto& layer : model["layers"]) {
      std::string type = layer["type"];
      if (type == "FullyConnected") {
        __fc(layers, layer);
      } else if (type == "Conv2D") {
        __conv(layers, layer);
      } else if (type == "AveragePooling") {
        __avg(layers, layer);
      } else if (type == "MaxPooling") {
        __max(layers, layer);
      } else if (type == "BatchNorm") {
        __batchnorm(layers, layer);
      } else if (type == "Dropout") {
        __dropout(layers, layer);
      } else if (type == "Flatten") {
        layers.push_back(std::make_shared<FlattenLayer>());
      } else if (type == "ReLU") {
        layers.push_back(std::make_shared<ReLULayer>());
      } else if (type == "Sigmoid") {
        layers.push_back(std::make_shared<SigmoidLayer>());
      } else if (type == "LReLU") {
        layers.push_back(std::make_shared<LReLULayer>());
      } else if (type == "Tanh") {
        layers.push_back(std::make_shared<TanhLayer>());
      } else if (type == "Softmax") {
        layers.push_back(std::make_shared<SoftmaxLayer>());
      } else {
        std::cerr << "Unknown layer type: " << type << std::endl;
      }
    }
  }

  static void _fc(json& layer, const FullyConnectedLayer* fc) {
    layer["numInput"] = fc->getNumInput();
    layer["numOutput"] = fc->getNumOutput();
    layer["weights"] = toStd2DVector(fc->getWeights());
    layer["biases"] = toStdVector(fc->getBiases());
  }

  static void __fc(std::vector<std::shared_ptr<BaseLayer>>& layers, const json& layer) {
    int numInput = layer["numInput"];
    int numOutput = layer["numOutput"];

    std::shared_ptr<FullyConnectedLayer> fc = std::make_shared<FullyConnectedLayer>(numInput, numOutput, INIT::NONE);

    std::vector<std::vector<val_t>> weights = layer["weights"].get<std::vector<std::vector<val_t>>>();
    fc->setWeights(toEigenMatrix(weights));

    std::vector<val_t> biases = layer["biases"].get<std::vector<val_t>>();
    fc->setBiases(toEigenVector(biases));

    layers.push_back(fc);
  }

  static void _conv(json& layer, const Conv2DLayer* conv) {
    layer["inChannels"] = conv->getInChannels();
    layer["outChannels"] = conv->getOutChannels();
    layer["kernelHeight"] = conv->getKernelHeight();
    layer["kernelWidth"] = conv->getKernelWidth();
    layer["stride"] = conv->getStride();
    layer["pad"] = conv->getPadding();

    layer["weights"] = toStd2DVector(conv->getWeights());
    layer["biases"] = toStdVector(conv->getBiases());
  }

  static void __conv(std::vector<std::shared_ptr<BaseLayer>>& layers, const json& layer) {
    size_t inChannels = layer["inChannels"];
    size_t outChannels = layer["outChannels"];
    size_t kH = layer["kernelHeight"];
    size_t kW = layer["kernelWidth"];
    size_t stride = layer["stride"];
    size_t pad = layer["pad"];

    std::shared_ptr<Conv2DLayer> conv = std::make_shared<Conv2DLayer>(inChannels, outChannels, kH, kW, stride, pad);

    std::vector<std::vector<val_t>> weights = layer["weights"].get<std::vector<std::vector<val_t>>>();
    conv->setWeights(toEigenMatrix(weights));

    std::vector<val_t> biases = layer["biases"].get<std::vector<val_t>>();
    conv->setBiases(toEigenVector(biases));

    layers.push_back(conv);
  }

  static void _avg(json& layer, const AveragePoolingLayer* avg) {
    layer["inputWidth"] = avg->getInputWidth();
    layer["inputHeight"] = avg->getInputHeight();
    layer["channels"] = avg->getChannels();
    layer["poolWidth"] = avg->getPoolWidth();
    layer["poolHeight"] = avg->getPoolHeight();
    layer["stride"] = avg->getStride();
    layer["pad"] = avg->getPadding();
  }

  static void __avg(std::vector<std::shared_ptr<BaseLayer>>& layers, const json& layer) {
    size_t inputWidth = layer["inputWidth"];
    size_t inputHeight = layer["inputHeight"];
    size_t channels = layer["channels"];
    size_t poolWidth = layer["poolWidth"];
    size_t poolHeight = layer["poolHeight"];
    size_t stride = layer["stride"];
    size_t pad = layer["pad"];

    layers.push_back(std::make_shared<AveragePoolingLayer>(BasePoolingLayer::with_input_dims, inputWidth, inputHeight,
        channels, poolWidth, poolHeight, stride, pad));
  }

  static void _max(json& layer, const MaxPoolingLayer* max) {
    layer["inputWidth"] = max->getInputWidth();
    layer["inputHeight"] = max->getInputHeight();
    layer["channels"] = max->getChannels();
    layer["poolWidth"] = max->getPoolWidth();
    layer["poolHeight"] = max->getPoolHeight();
    layer["stride"] = max->getStride();
    layer["pad"] = max->getPadding();
  }

  static void __max(std::vector<std::shared_ptr<BaseLayer>>& layers, const json& layer) {
    size_t inputWidth = layer["inputWidth"];
    size_t inputHeight = layer["inputHeight"];
    size_t channels = layer["channels"];
    size_t poolWidth = layer["poolWidth"];
    size_t poolHeight = layer["poolHeight"];
    size_t stride = layer["stride"];
    size_t pad = layer["pad"];

    layers.push_back(std::make_shared<MaxPoolingLayer>(BasePoolingLayer::with_input_dims, inputWidth, inputHeight,
        channels, poolWidth, poolHeight, stride, pad));
  }

  static void _batchnorm(json& layer, const BatchNormLayer* bn) {
    layer["numFeatures"] = bn->getNumFeatures();
    layer["gamma"] = toStdVector(bn->getGamma());
    layer["beta"] = toStdVector(bn->getBeta());
    layer["runningMean"] = toStdVector(bn->getRunningMean());
    layer["runningVar"] = toStdVector(bn->getRunningVar());
  }

  static void __batchnorm(std::vector<std::shared_ptr<BaseLayer>>& layers, const json& layer) {
    int numFeatures = layer["numFeatures"];
    std::shared_ptr<BatchNormLayer> bn = std::make_shared<BatchNormLayer>(numFeatures);

    std::vector<val_t> gamma = layer["gamma"].get<std::vector<val_t>>();
    bn->setGamma(toEigenVector(gamma));

    std::vector<val_t> beta = layer["beta"].get<std::vector<val_t>>();
    bn->setBeta(toEigenVector(beta));

    std::vector<val_t> runningMean = layer["runningMean"].get<std::vector<val_t>>();
    bn->setRunningMean(toEigenVector(runningMean));

    std::vector<val_t> runningVar = layer["runningVar"].get<std::vector<val_t>>();
    bn->setRunningVar(toEigenVector(runningVar));

    layers.push_back(bn);
  }

  // Dropout serialization
  static void _dropout(json& layer, const DropoutLayer* dropout) {
    layer["dropoutRate"] = dropout->getDropoutRate();
  }

  static void __dropout(std::vector<std::shared_ptr<BaseLayer>>& layers, const json& layer) {
    double dropoutRate = layer["dropoutRate"];
    layers.push_back(std::make_shared<DropoutLayer>(dropoutRate));
  }
};