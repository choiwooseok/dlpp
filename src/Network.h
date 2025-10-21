#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

#include "LossFunction.h"

#include "helper/EigenHelper.h"
#include "helper/Serializer.h"

#include "layers/base/BaseLayer.h"
#include "types.h"

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

  void backward(const vec_t &dY, double eta) {
    vec_t delta = dY;
    for (int i = layers.size() - 1; i >= 0; --i) {
      delta = layers[i]->backward(delta, eta);
    }
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

        backward(dY, eta);
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
      cout << "(" << idx << ") " << layer->getName() << endl;
      idx++;
    }
    cout << endl;
  }

  void save(const string &fileName) {
    json model = Serializer::marshal(layers);
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
    Serializer::unmarshal(layers, model);
  }

private:
  string _setTempDir(bool flag, steady_clock::time_point &tp) {
    if (!flag) {
      return "";
    }

    string dir = to_string(timePointToMillis(tp));
    filesystem::create_directories(BASE_DIR + dir);
    return dir;
  }

  void _autoSave(const string &dir, int epoch) {
    string time = to_string(getCurrentTimeMillis());
    save(dir + "/epoch_" + to_string(epoch + 1) + "_" + time + ".json");
  }

private:
  vector<shared_ptr<BaseLayer>> layers;
};