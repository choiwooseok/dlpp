#pragma once

#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "layers/base/BaseLayer.h"

#include "layers/FlattenLayer.h"
#include "layers/FullyConnectedLayer.h"

#include "layers/activations/LReLULayer.h"
#include "layers/activations/ReLULayer.h"
#include "layers/activations/SigmoidLayer.h"
#include "layers/activations/TanhLayer.h"

using json = nlohmann::json;

class Serializer {
public:
  static json marshal(vector<shared_ptr<BaseLayer>> &layers) {
    json model;
    for (const auto &layer : layers) {
      json layer_;
      layer_["type"] = layer->getName();

      if (layer->getName() == "FullyConnected") {
        _fc(layer_, static_cast<FullyConnectedLayer *>(layer.get()));
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
      } else {
        cerr << "Unknown layer type: " << type << endl;
      }
    }
  }

  static void _fc(json &layer, FullyConnectedLayer *fc) {
    layer["numInput"] = fc->getNumInput();
    layer["numOutput"] = fc->getNumOutput();
    layer["weights"] = fromEigenMatrix<val_t, mat_t>(fc->getWeights());
    layer["biases"] = fromEigenVector<val_t, vec_t>(fc->getBiases());
  }

  static void __fc(vector<shared_ptr<BaseLayer>> &layers, json &layer) {
    int numInput = layer["numInput"];
    int numOutput = layer["numOutput"];

    FullyConnectedLayer *fc = new FullyConnectedLayer(numInput, numOutput);

    vector<vector<val_t>> weights =
        layer["weights"].get<vector<vector<val_t>>>();
    fc->setWeights(toEigenMatrix<val_t, vector<vector<val_t>>>(weights));

    vector<val_t> biases = layer["biases"].get<vector<val_t>>();
    fc->setBiases(toEigenVector<val_t, vector<val_t>>(biases));

    layers.emplace_back(fc);
  }
};