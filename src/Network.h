#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

#include "types.h"

#include "LossFunction.h"

#include "layers/BaseLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/LReLULayer.h"
#include "layers/ReLULayer.h"
#include "layers/SigmoidLayer.h"

class Network {
public:
  void addLayer(BaseLayer *layer) { layers.emplace_back(layer); }

  vec_t forward(const vec_t &input) {
    vec_t output = input;
    for (const auto &layer : layers) {
      output = layer->forward(output);
    }
    return output;
  }

  template <typename LossFunction>
  void train(const tensor_t &input, const tensor_t &label, int epochs,
             double eta, bool verbose) {
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
      val_t L = val_t(0);

      for (int i = 0; i < input.size(); ++i) {
        vec_t Y = forward(input[i]);

        L += LossFunction::f(label[i], Y);
        vec_t dY = LossFunction::df(label[i], Y);

        for (int i = layers.size() - 1; i >= 0; --i) {
          dY = layers[i]->backward(dY, eta);
        }
      }

      if (verbose) {
        if ((epoch + 1) % 10 == 0) {
          auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::steady_clock::now() - begin);

          cout << "epoch: " << epoch + 1 << "/" << epochs
               << ", loss: " << L / input.size()
               << ", elapsed time: " << elapsed.count() << "s" << endl;
        }
      }
    }
  }

  void save(const string &filepath) {
    using json = nlohmann::ordered_json;

    json model;

    for (const auto &layer : layers) {
      json layer_json;
      layer_json["type"] = layer->getName();
      layer_json["numInput"] = layer->getNumInput();
      layer_json["numOutput"] = layer->getNumOutput();

      if (layer->getName() == "FullyConnected") {
        FullyConnectedLayer *fc =
            static_cast<FullyConnectedLayer *>(layer.get());
        layer_json["weights"] = fc->getWeights();
        layer_json["biases"] = fc->getBiases();
      }

      model["layers"].push_back(layer_json);
    }
    ofstream file(filepath);
    file.clear();
    file << model.dump(2);
    file.close();
  }

  void load(const string &filepath) {
    using json = nlohmann::json;

    ifstream file(filepath);
    json model;
    file >> model;
    file.close();

    for (auto &layer_json : model["layers"]) {
      string type = layer_json["type"];
      int numInput = layer_json["numInput"];
      int numOutput = layer_json["numOutput"];

      if (type == "FullyConnected") {
        auto weights = layer_json["weights"].get<vector<vec_t>>();
        auto biases = layer_json["biases"].get<vec_t>();

        FullyConnectedLayer *fcLayer =
            new FullyConnectedLayer(numInput, numOutput);

        fcLayer->setWeights(weights);
        fcLayer->setBiases(biases);

        layers.emplace_back(fcLayer);

      } else if (type == "ReLU") {
        layers.emplace_back(new ReLULayer(numInput));
      } else if (type == "SigmoidLayer") {
        layers.emplace_back(new SigmoidLayer(numInput));
      } else if (type == "LReLU") {
        layers.emplace_back(new LReLULayer(numInput));
      } else {
        cerr << "Unknown layer type: " << type << endl;
      }
    }
  }

private:
  vector<unique_ptr<BaseLayer>> layers;
};