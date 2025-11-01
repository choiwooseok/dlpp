#pragma once

#include "base/BaseLayer.h"
#include <random>

class DropoutLayer : public BaseLayer {
 public:
  explicit DropoutLayer(double dropoutRate = 0.5, bool isTraining = false)
      : BaseLayer("Dropout"),
        dropoutRate_(dropoutRate),
        keepProb_(1.0 - dropoutRate),
        isTraining_(isTraining) {
    if (dropoutRate < 0.0 || dropoutRate >= 1.0) {
      throw std::runtime_error("Dropout rate must be in [0, 1)");
    }

    // Initialize random number generator
    std::random_device rd;
    rng_.seed(rd());
  }

  virtual ~DropoutLayer() = default;

 public:
  tensor_t forward(const tensor_t &input) override {
    lastInput_ = input;

    if (!isTraining_ || dropoutRate_ == 0.0) {
      // Inference mode or no dropout: pass through
      return input;
    }

    // Training mode: apply dropout
    return _forwardTrain(input);
  }

  tensor_t backward(const tensor_t &dY) override {
    if (!isTraining_ || dropoutRate_ == 0.0) {
      // No dropout was applied: pass gradient through
      return dY;
    }

    // Apply dropout mask to gradients
    const size_t totalSize = dY.totalSize();
    tensor_t dX(dY.shape);

    const val_t scale = static_cast<val_t>(1.0 / keepProb_);

    for (size_t i = 0; i < totalSize; ++i) {
      dX[i] = dY[i] * mask_[i] * scale;
    }

    return dX;
  }

  void updateParams(Optimizer * /*optimizer*/) override {
    // Dropout has no trainable parameters
  }

  void info() override {
    cout << "[rate=" << dropoutRate_
         << " keep=" << keepProb_ << "]";
  }

  // Mode control
  void setTraining(bool training) { isTraining_ = training; }
  bool isTraining() const { return isTraining_; }

  // Getters and Setters
  double getDropoutRate() const { return dropoutRate_; }
  double getKeepProb() const { return keepProb_; }

  void setDropoutRate(double rate) {
    if (rate < 0.0 || rate >= 1.0) {
      throw std::runtime_error("Dropout rate must be in [0, 1)");
    }
    dropoutRate_ = rate;
    keepProb_ = 1.0 - rate;
  }

 private:
  tensor_t _forwardTrain(const tensor_t &input) {
    const size_t totalSize = input.totalSize();

    // Resize mask if needed
    if (mask_.size() != totalSize) {
      mask_.resize(totalSize);
    }

    // Generate dropout mask
    _generateMask(totalSize);

    // Apply mask with inverted dropout (scale by 1/keepProb)
    tensor_t output(input.shape);
    const val_t scale = static_cast<val_t>(1.0 / keepProb_);

    for (size_t i = 0; i < totalSize; ++i) {
      output[i] = input[i] * mask_[i] * scale;
    }

    return output;
  }

  void _generateMask(size_t size) {
    std::bernoulli_distribution dist(keepProb_);

    for (size_t i = 0; i < size; ++i) {
      mask_[i] = dist(rng_) ? 1.0f : 0.0f;
    }
  }

 private:
  // Configuration
  double dropoutRate_;  // Probability of dropping a neuron
  double keepProb_;     // Probability of keeping a neuron (1 - dropoutRate)
  bool isTraining_;

  // Dropout mask (1 = keep, 0 = drop)
  std::vector<val_t> mask_;

  // Random number generator
  mutable std::mt19937 rng_;

  // Cached for backward
  tensor_t lastInput_;
};