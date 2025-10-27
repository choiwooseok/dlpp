#pragma once

#include "types.h"
#include <memory>
#include <unordered_map>
#include <string>

// Base Optimizer Interface
class Optimizer {
 public:
  explicit Optimizer(double learningRate) : learningRate_(learningRate) {}
  virtual ~Optimizer() = default;

  // Update parameters given gradients
  // paramId: unique identifier for the parameter (for stateful optimizers)
  virtual void update(mat_t& param, const mat_t& grad, const std::string& paramId) = 0;
  virtual void update(vec_t& param, const vec_t& grad, const std::string& paramId) = 0;

  // Reset optimizer state (useful between training sessions)
  virtual void reset() = 0;

  double getLearningRate() const { return learningRate_; }
  void setLearningRate(double lr) { learningRate_ = lr; }

 protected:
  double learningRate_;
};

// Gradient Descent (Vanilla)
class GD : public Optimizer {
 public:
  explicit GD(double learningRate = 0.01) : Optimizer(learningRate) {}
  virtual ~GD() = default;

  void update(mat_t& param, const mat_t& grad, const std::string& paramId) override {
    param.noalias() -= static_cast<val_t>(learningRate_) * grad;
  }

  void update(vec_t& param, const vec_t& grad, const std::string& paramId) override {
    param.noalias() -= static_cast<val_t>(learningRate_) * grad;
  }

  // GD is stateless, nothing to reset
  void reset() override {}
};

// Stochastic Gradient Descent with Momentum
class SGD : public Optimizer {
 public:
  explicit SGD(double learningRate = 0.01, double momentum = 0.9)
      : Optimizer(learningRate), momentum_(momentum) {}
  virtual ~SGD() = default;

  void update(mat_t& param, const mat_t& grad, const std::string& paramId) override {
    // Get or create velocity
    auto& v = velocityMat_[paramId];
    if (v.rows() != grad.rows() || v.cols() != grad.cols()) {
      v = mat_t::Zero(grad.rows(), grad.cols());
    }

    // Update velocity: v = momentum * v + grad
    v = static_cast<val_t>(momentum_) * v + grad;

    // Update parameters: param = param - lr * v
    param.noalias() -= static_cast<val_t>(learningRate_) * v;
  }

  void update(vec_t& param, const vec_t& grad, const std::string& paramId) override {
    // Get or create velocity
    auto& v = velocityVec_[paramId];
    if (v.size() != grad.size()) {
      v = vec_t::Zero(grad.size());
    }

    // Update velocity: v = momentum * v + grad
    v = static_cast<val_t>(momentum_) * v + grad;

    // Update parameters: param = param - lr * v
    param.noalias() -= static_cast<val_t>(learningRate_) * v;
  }

  void reset() override {
    velocityMat_.clear();
    velocityVec_.clear();
  }

  double getMomentum() const { return momentum_; }
  void setMomentum(double momentum) { momentum_ = momentum; }

 private:
  double momentum_;
  std::unordered_map<std::string, mat_t> velocityMat_;
  std::unordered_map<std::string, vec_t> velocityVec_;
};

// Adam Optimizer
class Adam : public Optimizer {
 public:
  explicit Adam(double learningRate = 0.001,
                double beta1 = 0.9,
                double beta2 = 0.999,
                double epsilon = 1e-8)
      : Optimizer(learningRate),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        t_(0) {}
  virtual ~Adam() = default;

  void update(mat_t& param, const mat_t& grad, const std::string& paramId) override {
    // Increment timestep
    if (momentMat_[paramId].size() == 0) {
      t_++;
    }

    // Get or create first moment (mean)
    auto& m = momentMat_[paramId];
    if (m.rows() != grad.rows() || m.cols() != grad.cols()) {
      m = mat_t::Zero(grad.rows(), grad.cols());
    }

    // Get or create second moment (variance)
    auto& v = velocityMat_[paramId];
    if (v.rows() != grad.rows() || v.cols() != grad.cols()) {
      v = mat_t::Zero(grad.rows(), grad.cols());
    }

    // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
    m = static_cast<val_t>(beta1_) * m +
        static_cast<val_t>(1.0 - beta1_) * grad;

    // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
    v = static_cast<val_t>(beta2_) * v +
        static_cast<val_t>(1.0 - beta2_) * grad.array().square().matrix();

    // Bias correction
    const val_t mHat_factor = static_cast<val_t>(1.0 / (1.0 - std::pow(beta1_, t_)));
    const val_t vHat_factor = static_cast<val_t>(1.0 / (1.0 - std::pow(beta2_, t_)));

    mat_t mHat = mHat_factor * m;
    mat_t vHat = vHat_factor * v;

    // Update parameters: param = param - lr * mHat / (sqrt(vHat) + epsilon)
    // Use array operations for element-wise division and convert back to matrix
    param -= (static_cast<val_t>(learningRate_) *
              mHat.array() / (vHat.array().sqrt() + static_cast<val_t>(epsilon_)))
                 .matrix();
  }

  void update(vec_t& param, const vec_t& grad, const std::string& paramId) override {
    // Increment timestep
    if (momentVec_[paramId].size() == 0) {
      t_++;
    }

    // Get or create first moment (mean)
    auto& m = momentVec_[paramId];
    if (m.size() != grad.size()) {
      m = vec_t::Zero(grad.size());
    }

    // Get or create second moment (variance)
    auto& v = velocityVec_[paramId];
    if (v.size() != grad.size()) {
      v = vec_t::Zero(grad.size());
    }

    // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
    m = static_cast<val_t>(beta1_) * m +
        static_cast<val_t>(1.0 - beta1_) * grad;

    // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
    v = static_cast<val_t>(beta2_) * v +
        static_cast<val_t>(1.0 - beta2_) * grad.array().square().matrix();

    // Bias correction
    const val_t mHat_factor = static_cast<val_t>(1.0 / (1.0 - std::pow(beta1_, t_)));
    const val_t vHat_factor = static_cast<val_t>(1.0 / (1.0 - std::pow(beta2_, t_)));

    vec_t mHat = mHat_factor * m;
    vec_t vHat = vHat_factor * v;

    // Update parameters: param = param - lr * mHat / (sqrt(vHat) + epsilon)
    // Use array operations for element-wise division and convert back to vector
    param -= (static_cast<val_t>(learningRate_) *
              mHat.array() / (vHat.array().sqrt() + static_cast<val_t>(epsilon_)))
                 .matrix();
  }

  void reset() override {
    momentMat_.clear();
    velocityMat_.clear();
    momentVec_.clear();
    velocityVec_.clear();
    t_ = 0;
  }

  double getBeta1() const { return beta1_; }
  double getBeta2() const { return beta2_; }
  double getEpsilon() const { return epsilon_; }

  void setBeta1(double beta1) { beta1_ = beta1; }
  void setBeta2(double beta2) { beta2_ = beta2; }
  void setEpsilon(double epsilon) { epsilon_ = epsilon; }

 private:
  double beta1_;    // Exponential decay rate for first moment
  double beta2_;    // Exponential decay rate for second moment
  double epsilon_;  // Small constant for numerical stability
  size_t t_;        // Timestep

  // First moment (mean) estimates
  std::unordered_map<std::string, mat_t> momentMat_;
  std::unordered_map<std::string, vec_t> momentVec_;

  // Second moment (variance) estimates
  std::unordered_map<std::string, mat_t> velocityMat_;
  std::unordered_map<std::string, vec_t> velocityVec_;
};