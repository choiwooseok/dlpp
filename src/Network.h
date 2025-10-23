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

  tensor_t forward(const tensor_t &input) {
    tensor_t output = input;
    for (const auto &layer : layers) {
      output = layer->forward(output);
    }
    return output;
  }

  void backward(const tensor_t &dY, double eta) {
    tensor_t delta = dY;
    for (int i = (int)layers.size() - 1; i >= 0; --i) {
      delta = layers[i]->backward(delta);
      layers[i]->updateParams(eta);
    }
  }

  // mini-batch training
  template <typename LossFunction>
  void train(const tensor_t &input, const tensor_t &label, int epochs,
             double eta, size_t miniBatchSize = 1,
             bool shouldAutoSave = false) {
    steady_clock::time_point begin = steady_clock::now();

    size_t numSamples = input.shape[0];
    size_t numBatches = (numSamples + miniBatchSize - 1) / miniBatchSize;

    string tempDir = _setTempDir(shouldAutoSave, begin);

    for (int epoch = 0; epoch < epochs; ++epoch) {
      val_t L = val_t(0);

      for (size_t batch = 0; batch < numBatches; ++batch) {
        size_t start = batch * miniBatchSize;
        size_t currentBatchSize = min(miniBatchSize, numSamples - start);

        tensor_t in_batch = _getBatch(input, start, currentBatchSize);
        tensor_t label_batch = _getBatch(label, start, currentBatchSize);

        tensor_t Y = forward(in_batch);

        if ((epoch + 1) % 10 == 0) {
          L += LossFunction::f(label_batch, Y);
        }

        tensor_t dY = LossFunction::df(label_batch, Y);
        backward(dY, eta);
      }

      bool isSavePoint = (epoch + 1) % 100 == 0;
      if (shouldAutoSave && isSavePoint) {
        _autoSave(tempDir, epoch);
      }

      if ((epoch + 1) % 10 == 0) {
        auto elapsed = duration_cast<seconds>(steady_clock::now() - begin);
        cout << "epoch: " << epoch + 1 << "/" << epochs
             << ", loss: " << L / numSamples
             << ", elapsed time: " << elapsed.count() << "s" << endl;
      }
    }
  }

  void infos() {
    int idx = 0;
    for (const auto &layer : layers) {
      cout << "(" << idx << ") " << layer->getName() << " ";
      layer->info();
      cout << endl;
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

  // helper: build a mini-batch tensor_t from in starting at index start with
  // batchSize samples
  tensor_t _getBatch(const tensor_t &in, size_t start, size_t batchSize) {
    vector<size_t> outShape{batchSize};
    for (size_t i = 1; i < in.shape.size(); ++i)
      outShape.push_back(in.shape[i]);

    tensor_t out(outShape);

    size_t sampleSize = in.totalSize() / in.shape[0];

    for (size_t b = 0; b < batchSize; ++b) {
      size_t srcOffset = (start + b) * sampleSize;
      size_t dstOffset = b * sampleSize;
      for (size_t i = 0; i < sampleSize; ++i) {
        out.data[dstOffset + i] = in.data[srcOffset + i];
      }
    }

    return out;
  }

private:
  vector<shared_ptr<BaseLayer>> layers;
};