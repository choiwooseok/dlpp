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

#include "helper/EigenHelper.h"

using json = nlohmann::json;
using namespace std::chrono;

class Network {
private:
  inline static const string BASE_DIR = "../resource/model/";

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
             double eta, bool shouldAutoSave = false) {
    steady_clock::time_point begin = steady_clock::now();

    string tempDir = _setTempDir(shouldAutoSave, begin);

    for (int epoch = 0; epoch < epochs; ++epoch) {
      val_t L = val_t(0);

      for (int i = 0; i < input.rows(); ++i) {
        vec_t Y = forward(input.row(i));

        L += LossFunction::f(label.row(i), Y);
        vec_t dY = LossFunction::df(label.row(i), Y);

        for (int i = layers.size() - 1; i >= 0; --i) {
          dY = layers[i]->backward(dY, eta);
        }
      }

      if ((epoch + 1) % 10 == 0) {
        auto elapsed = duration_cast<seconds>(steady_clock::now() - begin);

        cout << "epoch: " << epoch + 1 << "/" << epochs
             << ", loss: " << L / input.rows()
             << ", elapsed time: " << elapsed.count() << "s" << endl;
      }

      bool isSavePoint = (epoch + 1) % 100 == 0;
      if (shouldAutoSave && isSavePoint) {
        _autoSave(tempDir, epoch);
      }
    }
  }

  void infos() {
    int idx = 0;
    for (const auto &layer : layers) {
      cout << "(" << idx << "): " << layer->getName() << "("
           << layer->getNumInput() << ", " << layer->getNumOutput() << ")"
           << endl;
      idx++;
    }
    cout << endl;
  }

  void save(const string &fileName) {
    json model = to_json();
    ofstream file(BASE_DIR + fileName);
    file.clear();
    file << model.dump(2);
    file.close();
  }

  void load(const string &fileName) {
    ifstream file(BASE_DIR + fileName);
    json model;
    file >> model;
    file.close();
    from_json(model);
  }

private:
  json to_json() {
    json model;

    for (const auto &layer : layers) {
      json layer_json;
      layer_json["type"] = layer->getName();
      layer_json["numInput"] = layer->getNumInput();
      layer_json["numOutput"] = layer->getNumOutput();

      if (layer->getName() == "FullyConnected") {
        FullyConnectedLayer *fc =
            static_cast<FullyConnectedLayer *>(layer.get());
        layer_json["weights"] =
            fromEigenMatrix<val_t, tensor_t>(fc->getWeights());
        layer_json["biases"] = fromEigenVector<val_t, vec_t>(fc->getBiases());
      }

      model["layers"].push_back(layer_json);
    }
    return model;
  }

  void from_json(json model) {
    for (auto &layer_json : model["layers"]) {
      string type = layer_json["type"];
      int numInput = layer_json["numInput"];
      int numOutput = layer_json["numOutput"];

      if (type == "FullyConnected") {
        vector<vector<val_t>> weights =
            layer_json["weights"].get<vector<vector<val_t>>>();
        vector<val_t> biases = layer_json["biases"].get<vector<val_t>>();

        FullyConnectedLayer *fcLayer =
            new FullyConnectedLayer(numInput, numOutput);

        fcLayer->setWeights(
            toEigenMatrix<val_t, vector<vector<val_t>>>(weights));
        fcLayer->setBiases(toEigenVector<val_t, vector<val_t>>(biases));

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

  string _setTempDir(bool flag, steady_clock::time_point &tp) {
    if (!flag) {
      return "";
    }

    string dir =
        to_string(duration_cast<milliseconds>(tp.time_since_epoch()).count());
    filesystem::create_directories(BASE_DIR + dir);
    return dir;
  }

  void _autoSave(const string &dir, int epoch) {
    save(dir + "/epoch_" + to_string(epoch + 1) + "_" +
         to_string(
             duration_cast<milliseconds>(system_clock::now().time_since_epoch())
                 .count()) +
         ".json");
  }

private:
  vector<unique_ptr<BaseLayer>> layers;
};