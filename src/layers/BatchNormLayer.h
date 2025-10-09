#pragma once

#include "base/BaseLayer.h"

class BatchNormLayer : public BaseLayer {
 public:
  explicit BatchNormLayer(int numFeatures, double momentum = 0.9, double epsilon = 1e-5, bool isTraining = false)
      : BaseLayer("BatchNorm"),
        numFeatures_(numFeatures),
        momentum_(momentum),
        epsilon_(epsilon),
        isTraining_(isTraining),
        accumSteps_(0) {
    _initializeParameters();
  }

  virtual ~BatchNormLayer() = default;

  Tensor forward(const Tensor& input) override {
    lastInput_ = input;

    if (isTraining_) {
      return _forwardTrain(input);
    } else {
      return _forwardInference(input);
    }
  }

  Tensor backward(const Tensor& dY) override {
    const auto dims = _extractDims(lastInput_);
    if (dims.batchSize == 0) {
      return Tensor(lastInput_.shape());
    }

    // Accumulate parameter gradients
    _accumulateGradients(dY, dims);

    // Compute input gradients
    return _backwardCompute(dY, dims);
  }

  void updateParams(Optimizer* optimizer) override {
    if (accumSteps_ == 0 || optimizer == nullptr) {
      return;
    }

    // Average the accumulated gradients
    vec_t avgDGamma = dGamma_accum_ / static_cast<val_t>(accumSteps_);
    vec_t avgDBeta = dBeta_accum_ / static_cast<val_t>(accumSteps_);

    // Use optimizer to update parameters
    optimizer->update(gamma_, avgDGamma, layerId_ + "_gamma");
    optimizer->update(beta_, avgDBeta, layerId_ + "_beta");

    dGamma_accum_.setZero();
    dBeta_accum_.setZero();
    accumSteps_ = 0;
  }

  void info() override {
    std::cout << std::format("[features={} momentum={} eps={}]", numFeatures_, momentum_, epsilon_);
  }

 public:
  // Mode control
  void setTraining(bool training) override {
    isTraining_ = training;
  }
  bool isTraining() const {
    return isTraining_;
  }

  // Getters and Setters
  int getNumFeatures() const {
    return numFeatures_;
  }
  const vec_t& getGamma() const {
    return gamma_;
  }
  const vec_t& getBeta() const {
    return beta_;
  }
  const vec_t& getRunningMean() const {
    return runningMean_;
  }
  const vec_t& getRunningVar() const {
    return runningVar_;
  }

  void setGamma(const vec_t& gamma) {
    gamma_ = gamma;
  }
  void setBeta(const vec_t& beta) {
    beta_ = beta;
  }
  void setRunningMean(const vec_t& mean) {
    runningMean_ = mean;
  }
  void setRunningVar(const vec_t& var) {
    runningVar_ = var;
  }

 private:
  struct Dimensions {
    size_t batchSize;
    size_t spatialSize;      // H * W for 4D, 1 for 2D
    size_t totalPerChannel;  // batchSize * spatialSize
  };

  Dimensions _extractDims(const Tensor& input) const {
    Dimensions dims;

    if (input.dim() == 4) {
      // Convolutional: (N, C, H, W)
      dims.batchSize = input.shape(0);
      dims.spatialSize = input.shape(2) * input.shape(3);
    } else if (input.dim() == 2) {
      // Fully Connected: (N, C)
      dims.batchSize = input.shape(0);
      dims.spatialSize = 1;
    } else {
      throw LayerException(getName(), "unsupported input dimensions");
    }

    dims.totalPerChannel = dims.batchSize * dims.spatialSize;
    return dims;
  }

  Tensor _forwardTrain(const Tensor& input) {
    const auto dims = _extractDims(input);
    Tensor output(input.shape());

    // Compute batch statistics per channel
    _computeBatchStatistics(input, dims);

    // Normalize and scale
    _normalizeAndScale(input, output, batchMean_, batchVar_, dims);

    // Update running statistics
    _updateRunningStatistics();

    return output;
  }

  Tensor _forwardInference(const Tensor& input) const {
    const auto dims = _extractDims(input);
    Tensor output(input.shape());

    // Use running statistics for normalization
    _normalizeAndScale(input, output, runningMean_, runningVar_, dims);

    return output;
  }

  void _computeBatchStatistics(const Tensor& input, const Dimensions& dims) {
    batchMean_.setZero();
    batchVar_.setZero();

    const val_t* inData = input.data();

    // Compute mean for each channel
    for (int c = 0; c < numFeatures_; ++c) {
      val_t sum = 0.0;

      for (size_t n = 0; n < dims.batchSize; ++n) {
        for (size_t s = 0; s < dims.spatialSize; ++s) {
          const size_t idx = _getIndex(n, c, s, dims);
          sum += inData[idx];
        }
      }

      batchMean_[c] = sum / static_cast<val_t>(dims.totalPerChannel);
    }

    // Compute variance for each channel
    for (int c = 0; c < numFeatures_; ++c) {
      val_t sumSq = 0.0;

      for (size_t n = 0; n < dims.batchSize; ++n) {
        for (size_t s = 0; s < dims.spatialSize; ++s) {
          const size_t idx = _getIndex(n, c, s, dims);
          const val_t diff = inData[idx] - batchMean_[c];
          sumSq += diff * diff;
        }
      }

      batchVar_[c] = sumSq / static_cast<val_t>(dims.totalPerChannel);
    }
  }

  void _normalizeAndScale(const Tensor& input, Tensor& output, const vec_t& mean, const vec_t& var,
      const Dimensions& dims) const {
    const val_t* inData = input.data();
    val_t* outData = output.data();

    for (int c = 0; c < numFeatures_; ++c) {
      const val_t invStd = 1.0 / std::sqrt(var[c] + static_cast<val_t>(epsilon_));
      const val_t scale = gamma_[c];
      const val_t shift = beta_[c];

      for (size_t n = 0; n < dims.batchSize; ++n) {
        for (size_t s = 0; s < dims.spatialSize; ++s) {
          const size_t idx = _getIndex(n, c, s, dims);
          // Normalize: (x - mean) / std
          const val_t normalized = (inData[idx] - mean[c]) * invStd;
          // Scale and shift: gamma * normalized + beta
          outData[idx] = scale * normalized + shift;
        }
      }
    }
  }

  void _updateRunningStatistics() {
    const val_t momentum = static_cast<val_t>(momentum_);
    const val_t oneMinusMomentum = 1.0 - momentum;

    for (int c = 0; c < numFeatures_; ++c) {
      runningMean_[c] = momentum * runningMean_[c] + oneMinusMomentum * batchMean_[c];
      runningVar_[c] = momentum * runningVar_[c] + oneMinusMomentum * batchVar_[c];
    }
  }

  void _accumulateGradients(const Tensor& dY, const Dimensions& dims) {
    const val_t* dYdata = dY.data();
    const val_t* inData = lastInput_.data();

    for (int c = 0; c < numFeatures_; ++c) {
      val_t dGamma = 0.0;
      val_t dBeta = 0.0;

      const val_t invStd = 1.0 / std::sqrt(batchVar_[c] + static_cast<val_t>(epsilon_));

      for (size_t n = 0; n < dims.batchSize; ++n) {
        for (size_t s = 0; s < dims.spatialSize; ++s) {
          const size_t idx = _getIndex(n, c, s, dims);
          const val_t normalized = (inData[idx] - batchMean_[c]) * invStd;

          dGamma += dYdata[idx] * normalized;
          dBeta += dYdata[idx];
        }
      }

      dGamma_accum_[c] += dGamma;
      dBeta_accum_[c] += dBeta;
    }

    accumSteps_ += dims.batchSize;
  }

  Tensor _backwardCompute(const Tensor& dY, const Dimensions& dims) const {
    Tensor dX(lastInput_.shape());

    const val_t* dYdata = dY.data();
    const val_t* inData = lastInput_.data();
    val_t* dXdata = dX.data();

    const val_t invN = 1.0 / static_cast<val_t>(dims.totalPerChannel);

    for (int c = 0; c < numFeatures_; ++c) {
      const val_t invStd = 1.0 / std::sqrt(batchVar_[c] + static_cast<val_t>(epsilon_));
      const val_t scale = gamma_[c];

      // Compute gradient components
      val_t dMean = 0.0;
      val_t dVar = 0.0;

      for (size_t n = 0; n < dims.batchSize; ++n) {
        for (size_t s = 0; s < dims.spatialSize; ++s) {
          const size_t idx = _getIndex(n, c, s, dims);

          dVar += dYdata[idx] * scale * (inData[idx] - batchMean_[c]) * (-0.5) * std::pow(invStd, 3);
          dMean += dYdata[idx] * scale * (-invStd);
        }
      }

      dMean += dVar * (-2.0) * invN * std::accumulate(inData, inData + lastInput_.size(), 0.0) /
               static_cast<val_t>(dims.totalPerChannel);

      // Compute input gradients
      for (size_t n = 0; n < dims.batchSize; ++n) {
        for (size_t s = 0; s < dims.spatialSize; ++s) {
          const size_t idx = _getIndex(n, c, s, dims);

          dXdata[idx] = dYdata[idx] * scale * invStd + dVar * 2.0 * (inData[idx] - batchMean_[c]) * invN + dMean * invN;
        }
      }
    }

    return dX;
  }

  size_t _getIndex(size_t n, size_t c, size_t s, const Dimensions& dims) const {
    if (lastInput_.dim() == 4) {
      // (N, C, H, W)
      const size_t h = s / lastInput_.shape(3);
      const size_t w = s % lastInput_.shape(3);
      return n * lastInput_.strides(0) + c * lastInput_.strides(1) + h * lastInput_.strides(2) + w;
    } else {
      // (N, C)
      return n * lastInput_.strides(0) + c;
    }
  }

  void _initializeParameters() {
    gamma_ = vec_t::Ones(numFeatures_);
    beta_ = vec_t::Zero(numFeatures_);
    runningMean_ = vec_t::Zero(numFeatures_);
    runningVar_ = vec_t::Ones(numFeatures_);

    dGamma_accum_ = vec_t::Zero(numFeatures_);
    dBeta_accum_ = vec_t::Zero(numFeatures_);

    batchMean_ = vec_t::Zero(numFeatures_);
    batchVar_ = vec_t::Ones(numFeatures_);
  }

 private:
  // Configuration
  int numFeatures_;
  double momentum_;
  double epsilon_;
  bool isTraining_;

  // Trainable parameters
  vec_t gamma_;  // Scale
  vec_t beta_;   // Shift

  // Running statistics (for inference)
  vec_t runningMean_;
  vec_t runningVar_;

  // Batch statistics (for training)
  vec_t batchMean_;
  vec_t batchVar_;

  // Gradient accumulators
  vec_t dGamma_accum_;
  vec_t dBeta_accum_;
  size_t accumSteps_;

  // Cached for backward
  Tensor lastInput_;
};