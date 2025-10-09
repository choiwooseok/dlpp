#pragma once

#include <algorithm>
#include <concepts>
#include <coroutine>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <ranges>

#include "LossFunction.h"
#include "ResourceManager.h"
#include "Serializer.h"
#include "layers/base/BaseLayer.h"
#include "types.h"

using json = nlohmann::json;
using namespace std::chrono;
namespace ranges = std::ranges;
namespace views = std::views;

// Concepts
template <typename T>
concept LossFunction = requires(T, const Tensor& a, const Tensor& b) {
  { T::f(a, b) } -> std::convertible_to<val_t>;
  { T::df(a, b) } -> std::same_as<Tensor>;
};

template <typename T>
concept HasClassInfo = requires {
  { T::NUM_CLASSES } -> std::convertible_to<int>;
  { T::classToString(typename T::Class{}) } -> std::convertible_to<std::string>;
};

// Batch Generator using Coroutines
template <typename T>
struct Generator {
  struct promise_type {
    T current_value;

    Generator get_return_object() {
      return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    std::suspend_always initial_suspend() {
      return {};
    }
    std::suspend_always final_suspend() noexcept {
      return {};
    }
    void return_void() {}
    void unhandled_exception() {
      std::terminate();
    }

    std::suspend_always yield_value(T value) {
      current_value = std::move(value);
      return {};
    }
  };

  std::coroutine_handle<promise_type> handle;

  explicit Generator(std::coroutine_handle<promise_type> h)
      : handle(h) {}
  ~Generator() {
    if (handle)
      handle.destroy();
  }

  // Move-only
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;
  Generator(Generator&& other) noexcept
      : handle(std::exchange(other.handle, {})) {}
  Generator& operator=(Generator&& other) noexcept {
    if (this != &other) {
      if (handle)
        handle.destroy();
      handle = std::exchange(other.handle, {});
    }
    return *this;
  }

  bool next() {
    if (!handle || handle.done())
      return false;
    handle.resume();
    return !handle.done();
  }

  T& value() {
    return handle.promise().current_value;
  }
  const T& value() const {
    return handle.promise().current_value;
  }
};

class Network {
 public:
  Network()
      : optimizer_(nullptr) {}
  ~Network() = default;

  void addLayer(std::unique_ptr<BaseLayer> layer) {
    layers_.emplace_back(std::move(layer));
  }

  // Forward prop
  Tensor forward(const Tensor& input) const {
    Tensor output = input;
    for_each(layers_.begin(), layers_.end(), [&](const auto& layer) { output = layer->forward(output); });
    return output;
  }

  // Backprop with parameter updates
  void backward(const Tensor& dY) {
    Tensor delta = dY;
    for_each(layers_.rbegin(), layers_.rend(), [&](const auto& layer) {
      delta = layer->backward(delta);
      layer->updateParams(optimizer_);
    });
  }

  // mini-batch training
  // input (N,C,H,W)
  template <LossFunction LossFunc>
  void train(
      const Tensor& train_data, const Tensor& train_label, int epochs, Optimizer* optimizer, size_t miniBatchSize = 1,
      std::invocable auto&& onEpochCallback = []() {}) {
    if (optimizer == nullptr) {
      throw std::runtime_error("Optimizer cannot be null");
    }

    _setOptimizer(optimizer);
    _setTrainingMode(true);

    const steady_clock::time_point begin = steady_clock::now();
    const size_t numSamples = train_data.shape(0);
    const size_t numBatches = (numSamples + miniBatchSize - 1) / miniBatchSize;

    std::cout << "Starting training..." << std::endl;

    for (auto epoch : views::iota(0, epochs)) {
      val_t totalLoss = val_t(0);

      size_t batchIdx = 0;
      auto gen = _generateBatches(train_data, train_label, miniBatchSize);

      // process batch
      while (gen.next()) {
        const auto& [input, label] = gen.value();

        // Forward pass
        const auto output = forward(input);

        // Compute loss
        totalLoss += LossFunc::f(label, output);

        // Backward pass
        Tensor dY = LossFunc::df(label, output);
        backward(dY);

        optimizer_->step();  // for CyclicLR or OneCycleLR
        ++batchIdx;

        // Progress output
        std::cout << std::format("\rEpoch {}/{} - lr {:.6f}, Batch {}/{}", epoch + 1, epochs,
                         optimizer_->getLearningRate(), batchIdx, numBatches)
                  << std::flush;
      }

      std::cout << std::endl;
      std::invoke(std::forward<decltype(onEpochCallback)>(onEpochCallback));

      // Epoch summary
      const auto elapsed = duration_cast<seconds>(steady_clock::now() - begin);
      std::cout << std::format("Epoch {} - Loss: {:.6f} | Time: {}s\n", epoch + 1,
          totalLoss / static_cast<val_t>(numSamples), elapsed.count());
    }

    _setTrainingMode(false);
    std::cout << "Training completed successfully!" << std::endl;
  }

  template <HasClassInfo Data>
  class Result {
   public:
    struct ClassMetrics {
      int correct{0};
      int total{0};

      double accuracy() const noexcept {
        return total > 0 ? static_cast<double>(correct) / total : 0.0;
      }
    };

    int numCorrect = 0;
    int numSamples = 0;
    std::array<ClassMetrics, Data::NUM_CLASSES> classMetrics{};
    mat_t confusionMatrix{mat_t::Zero(Data::NUM_CLASSES, Data::NUM_CLASSES)};

   public:
    double getAccuracy() {
      if (numSamples == 0)
        return 0;
      return numCorrect / static_cast<double>(numSamples);
    }

    void printAccuracy() {
      double acc = getAccuracy();
      std::cout << std::string(60, '-') << std::endl;
      std::cout << std::format("Overall Accuracy: {:.2f}%", (acc * 100.0)) << std::endl;
      std::cout << std::string(60, '-') << std::endl;
    }

    void printPerClassAccuracy() {
      std::cout << std::string(60, '-') << std::endl;
      std::cout << "Per-Class Accuracy:" << std::endl;
      std::cout << std::string(60, '-') << std::endl;

      for (auto i : views::iota(0, Data::NUM_CLASSES)) {
        const auto& metrics = classMetrics[i];
        std::cout << std::format("{:<12}: {:.2f}% ({}/{})\n", Data::classToString(static_cast<typename Data::Class>(i)),
            metrics.accuracy() * 100.0, metrics.correct, metrics.total);
      }

      std::cout << std::endl;
    }

    void printConfusionMatrix() {
      std::cout << std::string(60, '-') << std::endl;
      std::cout << "Confusion Matrix (rows: true, cols: predicted):" << std::endl;
      std::cout << std::string(60, '-') << std::endl;

      for (auto i : views::iota(0, Data::NUM_CLASSES)) {
        std::cout << std::format("{:<12}:", Data::classToString(static_cast<typename Data::Class>(i)));
        for (auto j : views::iota(0, Data::NUM_CLASSES)) {
          std::cout << std::format("{:>5}", static_cast<int>(confusionMatrix(i, j)));
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  };

  template <HasClassInfo Data>
  Result<Data> test(
      const Tensor& test_data, const Tensor& test_label,
      std::invocable<int, int> auto&& wrongPredCallback = [](int /* idx */, int /* predicted */) {}) {
    _setTrainingMode(false);

    Result<Data> ret;
    ret.numSamples = test_data.shape(0);

    size_t batchIdx = 0;
    auto gen = _generateBatches(test_data, test_label, 1);

    while (gen.next()) {
      const auto& [input, label] = gen.value();

      Tensor Y = forward(input);

      int predicted = Y.max_element_idx();
      int expected = label.max_element_idx();

      auto& metrics = ret.classMetrics[expected];
      metrics.total++;
      ret.confusionMatrix(expected, predicted)++;

      if (predicted == expected) {
        ret.numCorrect++;
        metrics.correct++;
      } else {
        std::invoke(std::forward<decltype(wrongPredCallback)>(wrongPredCallback), batchIdx, predicted);
      }

      ++batchIdx;
      std::cout << std::format("\r - Test Progress {}/{}", batchIdx, ret.numSamples) << std::flush;
    }
    std::cout << "\n - Test Results:" << std::endl;
    ret.printPerClassAccuracy();
    ret.printConfusionMatrix();
    ret.printAccuracy();

    _setTrainingMode(true);

    return ret;
  }

  void infos() const {
    std::cout << "Network Architecture:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    for (size_t idx = 0; idx < layers_.size(); ++idx) {
      std::cout << std::format("({}) {} ", idx, layers_[idx]->getName());
      layers_[idx]->info();
      std::cout << std::endl;
    }

    std::cout << std::string(60, '=') << std::endl;
  }

  void save(const std::string& fileName) const {
    const auto fullPath = ResourceManager::instance().getModelPath(fileName);
    std::ofstream file(fullPath);

    if (!file.is_open()) {
      throw std::runtime_error(std::format("Failed to open file for saving: {}", fullPath.string()));
    }

    const json model = Serializer::marshal(layers_);
    file << model.dump(2);
    file.close();
    std::cout << std::format("Model saved as {}", fullPath.string()) << std::endl;
  }

  void load(const std::string& fileName) {
    const auto fullPath = ResourceManager::instance().getModelPath(fileName);
    std::ifstream file(fullPath);

    if (!file.is_open()) {
      throw std::runtime_error(std::format("Failed to open file for loading: {}", fullPath.string()));
    }

    json model;
    file >> model;
    file.close();

    // Clear existing layers
    layers_.clear();

    // Load new layers
    Serializer::unmarshal(layers_, model);

    std::cout << std::format("Model loaded from: {}", fullPath.string()) << std::endl;
    infos();
  }

 private:
  Tensor _batchShape(const Tensor& src, size_t miniBatchSize) {
    std::vector<size_t> shape = {miniBatchSize};
    for (auto i : views::iota(1UL, src.dim())) {
      shape.push_back(src.shape(i));
    }
    return Tensor(shape);
  }

  // batch extraction with in-place copy
  void _getBatchInPlace(const Tensor& src, Tensor& dst, size_t start, size_t batchSize) const {
    const size_t featureSize = src.size() / src.shape(0);
    const size_t offset = start * featureSize;
    const size_t copySize = batchSize * featureSize;

    std::copy_n(src.data() + offset, copySize, dst.data());
  }

  // Batch generator coroutine
  Generator<std::pair<Tensor, Tensor>> _generateBatches(const Tensor& data, const Tensor& labels, size_t batchSize) {
    const size_t numSamples = data.shape(0);

    for (size_t start = 0; start < numSamples; start += batchSize) {
      const size_t currentSize = std::min(batchSize, numSamples - start);

      Tensor batchData = _batchShape(data, currentSize);
      Tensor batchLabels = _batchShape(labels, currentSize);

      _getBatchInPlace(data, batchData, start, currentSize);
      _getBatchInPlace(labels, batchLabels, start, currentSize);

      co_yield {std::move(batchData), std::move(batchLabels)};
    }
  }

  void _setOptimizer(Optimizer* optimizer) {
    optimizer_ = optimizer;
  }

  void _setTrainingMode(bool training) {
    for (auto& layer : layers_) {
      layer->setTraining(training);
    }
  }

 private:
  std::vector<std::shared_ptr<BaseLayer>> layers_;
  Optimizer* optimizer_;
};
