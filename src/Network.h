#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>

#include "layers/base/BaseLayer.h"

#include "LossFunction.h"
#include "Serializer.h"
#include "types.h"

using json = nlohmann::json;
using namespace std::chrono;

class Network {
 public:
  Network() : optimizer_(nullptr) {}
  ~Network() = default;

  void addLayer(BaseLayer *layer) { layers_.emplace_back(layer); }

  void setOptimizer(Optimizer *optimizer) { optimizer_ = optimizer; }

  // Forward prop
  tensor_t forward(const tensor_t &input) const {
    tensor_t output = input;
    for (const auto &layer : layers_) {
      output = layer->forward(output);
    }
    return output;
  }

  // Backward prop with parameter updates
  void backward(const tensor_t &dY) {
    tensor_t delta = dY;
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
      delta = layers_[i]->backward(delta);
      layers_[i]->updateParams(optimizer_);
    }
  }

  // Optimized mini-batch training
  // input (N,C,H,W)
  // checkPoints -> (epoch + 1) % checkPoints == 0 then save;
  template <typename LossFunction>
  void train(const tensor_t &train_data,
             const tensor_t &train_label,
             int epochs,
             Optimizer *optimizer,
             size_t miniBatchSize = 1,
             size_t checkPoints = 0) {
    if (optimizer == nullptr) {
      throw std::runtime_error("Optimizer cannot be null");
    }

    setOptimizer(optimizer);

    const steady_clock::time_point begin = steady_clock::now();
    const size_t numSamples = train_data.shape[0];
    const size_t numBatches = (numSamples + miniBatchSize - 1) / miniBatchSize;

    const bool isCheckPointZero = checkPoints == 0;
    const string tempDir = isCheckPointZero ? "" : _setTempDir(begin);

    // Pre-allocate batch workspace
    _ensureBatchWorkspace(train_data, train_label, miniBatchSize);

    for (int epoch = 0; epoch < epochs; ++epoch) {
      val_t totalLoss = val_t(0);

      // Process each batch
      for (size_t batch = 0; batch < numBatches; ++batch) {
        const size_t start = batch * miniBatchSize;
        const size_t currentBatchSize = std::min(miniBatchSize, numSamples - start);

        // Get batch slices
        _getBatchInPlace(train_data, start, currentBatchSize, batchInput_);
        _getBatchInPlace(train_label, start, currentBatchSize, batchLabel_);

        // Forward pass
        tensor_t Y = forward(batchInput_);

        // Compute loss
        totalLoss += LossFunction::f(batchLabel_, Y);

        // Backward pass
        tensor_t dY = LossFunction::df(batchLabel_, Y);
        backward(dY);

        // Progress output
        if (batch % 10 == 0 || batch == numBatches - 1) {
          cout << "\rEpoch " << epoch + 1 << "/" << epochs << " - Batch " << batch + 1 << "/" << numBatches << std::flush;
        }
      }
      cout << endl;

      // checkpoint
      if (!isCheckPointZero && (epoch + 1) % checkPoints == 0) {
        _saveCheckPoints(tempDir, epoch);
      }

      // Epoch summary
      const auto elapsed = duration_cast<seconds>(steady_clock::now() - begin);
      cout << "Loss: " << totalLoss / static_cast<val_t>(numSamples) << " | Time: " << elapsed.count() << "s" << endl;
    }
  }

  // Display network architecture
  void infos() const {
    cout << "Network Architecture:" << endl;
    cout << string(60, '=') << endl;

    for (size_t idx = 0; idx < layers_.size(); ++idx) {
      cout << "(" << idx << ") " << layers_[idx]->getName() << " ";
      layers_[idx]->info();
      cout << endl;
    }

    cout << string(60, '=') << endl;
  }

  // Save model to JSON
  void save(const string &fileName) const {
    const json model = Serializer::marshal(layers_);

    const string fullPath = BASE_DIR + fileName;
    ofstream file(fullPath);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for saving: " + fullPath);
    }

    file << model.dump(2);
    file.close();
  }

  // Load model from JSON
  void load(const string &fileName) {
    const string fullPath = BASE_DIR + fileName;
    ifstream file(fullPath);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file for loading: " + fullPath);
    }

    json model;
    file >> model;
    file.close();

    // Clear existing layers
    layers_.clear();

    // Load new layers
    Serializer::unmarshal(layers_, model);

    cout << "Model loaded from: " << fullPath << endl;
  }

 private:
  // Set temporary directory for checkpoint save
  string _setTempDir(const steady_clock::time_point &tp) const {
    const string dir = to_string(timePointToMillis(tp));
    filesystem::create_directories(BASE_DIR + dir);
    return dir;
  }

  void _saveCheckPoints(const string &dir, int epoch) const {
    const string time = to_string(getCurrentTimeMillis());
    const string fileName = dir + "/epoch_" + to_string(epoch + 1) + "_" + time + ".json";
    save(fileName);
  }

  // Pre-allocate batch workspace to avoid repeated allocations
  void _ensureBatchWorkspace(const tensor_t &input, const tensor_t &label, size_t miniBatchSize) {
    // Allocate workspace for input batch
    vector<size_t> inputShape = {miniBatchSize};
    for (size_t i = 1; i < input.shape.size(); ++i) {
      inputShape.push_back(input.shape[i]);
    }

    if (batchInput_.shape != inputShape) {
      batchInput_ = tensor_t(inputShape);
    }

    // Allocate workspace for label batch
    vector<size_t> labelShape = {miniBatchSize};
    for (size_t i = 1; i < label.shape.size(); ++i) {
      labelShape.push_back(label.shape[i]);
    }

    if (batchLabel_.shape != labelShape) {
      batchLabel_ = tensor_t(labelShape);
    }
  }

  // batch extraction with in-place copy
  void _getBatchInPlace(const tensor_t &source, size_t start, size_t batchSize, tensor_t &dest) const {
    const size_t sampleSize = source.totalSize() / source.shape[0];

    // Direct memory copy for better performance
    const val_t *srcPtr = source.data.data() + (start * sampleSize);
    val_t *dstPtr = dest.data.data();
    const size_t copySize = batchSize * sampleSize;

    // Use memcpy for contiguous memory
    std::memcpy(dstPtr, srcPtr, copySize * sizeof(val_t));
  }

 private:
  inline static const string BASE_DIR = "../resource/model/";

  vector<shared_ptr<BaseLayer>> layers_;
  Optimizer *optimizer_;

  // Workspace for batch processing (reused across iterations)
  tensor_t batchInput_;
  tensor_t batchLabel_;
};