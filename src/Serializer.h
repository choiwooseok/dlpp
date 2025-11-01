#pragma once

#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "layers/base/BaseLayer.h"

#include "layers/AveragePoolingLayer.h"
#include "layers/Conv2DLayer.h"
#include "layers/FlattenLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/MaxPoolingLayer.h"
#include "layers/BatchNormLayer.h"
#include "layers/DropoutLayer.h"

#include "layers/activations/LReLULayer.h"
#include "layers/activations/ReLULayer.h"
#include "layers/activations/SigmoidLayer.h"
#include "layers/activations/SoftmaxLayer.h"
#include "layers/activations/TanhLayer.h"

using json = nlohmann::json;

class Serializer {
 public:
  static json marshal(const vector<shared_ptr<BaseLayer>> &layers) {
    json model;
    for (const auto &layer : layers) {
      json layer_;
      layer_["type"] = layer->getName();

      if (layer->getName() == "FullyConnected") {
        _fc(layer_, static_cast<FullyConnectedLayer *>(layer.get()));
      } else if (layer->getName() == "Conv2D") {
        _conv(layer_, static_cast<Conv2DLayer *>(layer.get()));
      } else if (layer->getName() == "AveragePooling") {
        _avg(layer_, static_cast<AveragePoolingLayer *>(layer.get()));
      } else if (layer->getName() == "MaxPooling") {
        _max(layer_, static_cast<MaxPoolingLayer *>(layer.get()));
      } else if (layer->getName() == "BatchNorm") {
        _batchnorm(layer_, static_cast<BatchNormLayer *>(layer.get()));
      } else if (layer->getName() == "Dropout") {
        _dropout(layer_, static_cast<DropoutLayer *>(layer.get()));
      }
      model["layers"].push_back(layer_);
    }
    return model;
  }

  static void unmarshal(vector<shared_ptr<BaseLayer>> &layers, json &model) {
    for (auto &layer : model["layers"]) {
      string type = layer["type"];
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
        layers.emplace_back(new FlattenLayer());
      } else if (type == "ReLU") {
        layers.emplace_back(new ReLULayer());
      } else if (type == "Sigmoid") {
        layers.emplace_back(new SigmoidLayer());
      } else if (type == "LReLU") {
        layers.emplace_back(new LReLULayer());
      } else if (type == "Tanh") {
        layers.emplace_back(new TanhLayer());
      } else if (type == "Softmax") {
        layers.emplace_back(new SoftmaxLayer());
      } else {
        cerr << "Unknown layer type: " << type << endl;
      }
    }
  }

  static void _fc(json &layer, FullyConnectedLayer *fc) {
    layer["numInput"] = fc->getNumInput();
    layer["numOutput"] = fc->getNumOutput();
    layer["weights"] = toStd2DVector(fc->getWeights());
    layer["biases"] = toStd1DVector(fc->getBiases());
  }

  static void __fc(vector<shared_ptr<BaseLayer>> &layers, json &layer) {
    int numInput = layer["numInput"];
    int numOutput = layer["numOutput"];

    FullyConnectedLayer *fc = new FullyConnectedLayer(numInput, numOutput);

    vector<vector<val_t>> weights = layer["weights"].get<vector<vector<val_t>>>();
    fc->setWeights(toEigenMatrix(weights));

    vector<val_t> biases = layer["biases"].get<vector<val_t>>();
    fc->setBiases(toEigenVector(biases));

    layers.emplace_back(fc);
  }

  static void _conv(json &layer, Conv2DLayer *conv) {
    layer["inChannels"] = conv->getInChannels();
    layer["outChannels"] = conv->getOutChannels();
    layer["kernelHeight"] = conv->getKernelHeight();
    layer["kernelWidth"] = conv->getKernelWidth();
    layer["stride"] = conv->getStride();
    layer["pad"] = conv->getPadding();

    layer["weights"] = toStd2DVector(conv->getWeights());
    layer["biases"] = toStd1DVector(conv->getBiases());
  }

  static void __conv(vector<shared_ptr<BaseLayer>> &layers, json &layer) {
    size_t inChannels = layer["inChannels"];
    size_t outChannels = layer["outChannels"];
    size_t kH = layer["kernelHeight"];
    size_t kW = layer["kernelWidth"];
    size_t stride = layer["stride"];
    size_t pad = layer["pad"];

    Conv2DLayer *conv = new Conv2DLayer(inChannels, outChannels, kH, kW, stride, pad);

    vector<vector<val_t>> weights = layer["weights"].get<vector<vector<val_t>>>();
    conv->setWeights(toEigenMatrix(weights));

    vector<val_t> biases = layer["biases"].get<vector<val_t>>();
    conv->setBiases(toEigenVector(biases));

    layers.emplace_back(conv);
  }

  static void _avg(json &layer, AveragePoolingLayer *avg) {
    layer["inputWidth"] = avg->getInputWidth();
    layer["inputHeight"] = avg->getInputHeight();
    layer["channels"] = avg->getChannels();
    layer["poolWidth"] = avg->getPoolWidth();
    layer["poolHeight"] = avg->getPoolHeight();
    layer["stride"] = avg->getStride();
    layer["pad"] = avg->getPadding();
  }

  static void __avg(vector<shared_ptr<BaseLayer>> &layers, json &layer) {
    size_t inputWidth = layer["inputWidth"];
    size_t inputHeight = layer["inputHeight"];
    size_t channels = layer["channels"];
    size_t poolWidth = layer["poolWidth"];
    size_t poolHeight = layer["poolHeight"];
    size_t stride = layer["stride"];
    size_t pad = layer["pad"];

    AveragePoolingLayer *avg = new AveragePoolingLayer(BasePoolingLayer::with_input_dims, inputWidth, inputHeight, channels, poolWidth, poolHeight, stride, pad);

    layers.emplace_back(avg);
  }

  static void _max(json &layer, MaxPoolingLayer *max) {
    layer["inputWidth"] = max->getInputWidth();
    layer["inputHeight"] = max->getInputHeight();
    layer["channels"] = max->getChannels();
    layer["poolWidth"] = max->getPoolWidth();
    layer["poolHeight"] = max->getPoolHeight();
    layer["stride"] = max->getStride();
    layer["pad"] = max->getPadding();
  }

  static void __max(vector<shared_ptr<BaseLayer>> &layers, json &layer) {
    size_t inputWidth = layer["inputWidth"];
    size_t inputHeight = layer["inputHeight"];
    size_t channels = layer["channels"];
    size_t poolWidth = layer["poolWidth"];
    size_t poolHeight = layer["poolHeight"];
    size_t stride = layer["stride"];
    size_t pad = layer["pad"];

    MaxPoolingLayer *max = new MaxPoolingLayer(BasePoolingLayer::with_input_dims, inputWidth, inputHeight, channels, poolWidth, poolHeight, stride, pad);

    layers.emplace_back(max);
  }

  static void _batchnorm(json &layer, BatchNormLayer *bn) {
    layer["numFeatures"] = bn->getNumFeatures();
    layer["gamma"] = toStd1DVector(bn->getGamma());
    layer["beta"] = toStd1DVector(bn->getBeta());
    layer["runningMean"] = toStd1DVector(bn->getRunningMean());
    layer["runningVar"] = toStd1DVector(bn->getRunningVar());
  }

  static void __batchnorm(vector<shared_ptr<BaseLayer>> &layers, json &layer) {
    int numFeatures = layer["numFeatures"];
    BatchNormLayer *bn = new BatchNormLayer(numFeatures);

    vector<val_t> gamma = layer["gamma"].get<vector<val_t>>();
    bn->setGamma(toEigenVector(gamma));

    vector<val_t> beta = layer["beta"].get<vector<val_t>>();
    bn->setBeta(toEigenVector(beta));

    vector<val_t> runningMean = layer["runningMean"].get<vector<val_t>>();
    bn->setRunningMean(toEigenVector(runningMean));

    vector<val_t> runningVar = layer["runningVar"].get<vector<val_t>>();
    bn->setRunningVar(toEigenVector(runningVar));

    layers.emplace_back(bn);
  }

  // Dropout serialization
  static void _dropout(json &layer, DropoutLayer *dropout) {
    layer["dropoutRate"] = dropout->getDropoutRate();
  }

  static void __dropout(vector<shared_ptr<BaseLayer>> &layers, json &layer) {
    double dropoutRate = layer["dropoutRate"];
    DropoutLayer *dropout = new DropoutLayer(dropoutRate);
    layers.emplace_back(dropout);
  }
};