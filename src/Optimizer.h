#pragma once

#include <cmath>
#include <string>
#include <unordered_map>

#include "types.h"

// Base Optimizer Interface
class Optimizer {
 public:
  explicit Optimizer(double learningRate)
      : learningRate_(learningRate) {}
  virtual ~Optimizer() = default;

  // Update parameters given gradients
  // paramId: unique identifier for the parameter (for stateful optimizers)
  virtual void update(mat_t& param, const mat_t& grad, const std::string& paramId) = 0;
  virtual void update(vec_t& param, const vec_t& grad, const std::string& paramId) = 0;

  // Reset optimizer state (useful between training sessions)
  virtual void reset() = 0;
  virtual void step() {}  // for CyclicLR or OneCycleLR

  virtual double getLearningRate() const {
    return learningRate_;
  }
  void setLearningRate(double lr) {
    learningRate_ = lr;
  }

 protected:
  double learningRate_;
};

// Gradient Descent (Vanilla)
class GD : public Optimizer {
 public:
  explicit GD(double learningRate = 0.01)
      : Optimizer(learningRate) {}
  virtual ~GD() = default;

  void update(mat_t& param, const mat_t& grad, const std::string& paramId) override {
    param.noalias() -= static_cast<val_t>(learningRate_) * grad;
  }

  void update(vec_t& param, const vec_t& grad, const std::string& paramId) override {
    param.noalias() -= static_cast<val_t>(learningRate_) * grad;
  }

  void reset() override {}
};

// Stochastic Gradient Descent with Momentum and Nesterov support
class SGD : public Optimizer {
 public:
  explicit SGD(double learningRate = 0.01, double momentum = 0.9, bool nesterov = false, double weightDecay = 0.0)
      : Optimizer(learningRate), momentum_(momentum), nesterov_(nesterov), weightDecay_(weightDecay) {}

  virtual ~SGD() = default;

  void update(mat_t& param, const mat_t& grad, const std::string& paramId) override {
    // Get or create velocity buffer
    auto& v = velocityMat_[paramId];
    if (v.rows() != grad.rows() || v.cols() != grad.cols()) {
      v = mat_t::Zero(grad.rows(), grad.cols());
    }

    // Apply weight decay if specified (L2 regularization)
    mat_t effectiveGrad = grad;
    if (weightDecay_ > 0.0) {
      effectiveGrad += static_cast<val_t>(weightDecay_) * param;
    }

    // Update velocity: v = momentum * v + grad
    v = static_cast<val_t>(momentum_) * v + effectiveGrad;

    // Update parameters
    if (nesterov_) {
      // Nesterov momentum: param = param - lr * (momentum * v + grad)
      param.noalias() -= static_cast<val_t>(learningRate_) * (static_cast<val_t>(momentum_) * v + effectiveGrad);
    } else {
      // Standard momentum: param = param - lr * v
      param.noalias() -= static_cast<val_t>(learningRate_) * v;
    }
  }

  void update(vec_t& param, const vec_t& grad, const std::string& paramId) override {
    // Get or create velocity buffer
    auto& v = velocityVec_[paramId];
    if (v.size() != grad.size()) {
      v = vec_t::Zero(grad.size());
    }

    // Apply weight decay if specified
    vec_t effectiveGrad = grad;
    if (weightDecay_ > 0.0) {
      effectiveGrad += static_cast<val_t>(weightDecay_) * param;
    }

    // Update velocity: v = momentum * v + grad
    v = static_cast<val_t>(momentum_) * v + effectiveGrad;

    // Update parameters
    if (nesterov_) {
      // Nesterov momentum
      param.noalias() -= static_cast<val_t>(learningRate_) * (static_cast<val_t>(momentum_) * v + effectiveGrad);
    } else {
      // Standard momentum
      param.noalias() -= static_cast<val_t>(learningRate_) * v;
    }
  }

  void reset() override {
    velocityMat_.clear();
    velocityVec_.clear();
  }

  // Getters and setters
  double getMomentum() const {
    return momentum_;
  }
  void setMomentum(double momentum) {
    momentum_ = momentum;
  }

  bool isNesterov() const {
    return nesterov_;
  }
  void setNesterov(bool nesterov) {
    nesterov_ = nesterov;
  }

  double getWeightDecay() const {
    return weightDecay_;
  }
  void setWeightDecay(double decay) {
    weightDecay_ = decay;
  }

 private:
  double momentum_;
  bool nesterov_;
  double weightDecay_;

  std::unordered_map<std::string, mat_t> velocityMat_;
  std::unordered_map<std::string, vec_t> velocityVec_;
};

// Adam Optimizer (Adaptive Moment Estimation)
class Adam : public Optimizer {
 public:
  explicit Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,
      double weightDecay = 0.0, bool amsgrad = false)
      : Optimizer(learningRate),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        weightDecay_(weightDecay),
        amsgrad_(amsgrad) {}

  virtual ~Adam() = default;

  void update(mat_t& param, const mat_t& grad, const std::string& paramId) override {
    // Increment parameter-specific timestep
    size_t& t = timestepMat_[paramId];
    t++;

    // Get or create first moment (mean) buffer
    auto& m = momentMat_[paramId];
    if (m.rows() != grad.rows() || m.cols() != grad.cols()) {
      m = mat_t::Zero(grad.rows(), grad.cols());
    }

    // Get or create second moment (variance) buffer
    auto& v = varianceMat_[paramId];
    if (v.rows() != grad.rows() || v.cols() != grad.cols()) {
      v = mat_t::Zero(grad.rows(), grad.cols());
    }

    // Apply weight decay if specified (AdamW variant)
    mat_t effectiveGrad = grad;
    if (weightDecay_ > 0.0) {
      effectiveGrad += static_cast<val_t>(weightDecay_) * param;
    }

    // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
    m = static_cast<val_t>(beta1_) * m + static_cast<val_t>(1.0 - beta1_) * effectiveGrad;

    // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
    v = static_cast<val_t>(beta2_) * v + static_cast<val_t>(1.0 - beta2_) * effectiveGrad.array().square().matrix();

    // Compute bias-corrected first moment estimate
    const val_t bias1Correction = static_cast<val_t>(1.0 / (1.0 - std::pow(beta1_, t)));
    mat_t mHat = bias1Correction * m;

    // Compute bias-corrected second raw moment estimate
    const val_t bias2Correction = static_cast<val_t>(1.0 / (1.0 - std::pow(beta2_, t)));
    mat_t vHat = bias2Correction * v;

    // AMSGrad variant (maintains max of past second moments)
    if (amsgrad_) {
      auto& vMax = vMaxMat_[paramId];
      if (vMax.rows() != grad.rows() || vMax.cols() != grad.cols()) {
        vMax = mat_t::Zero(grad.rows(), grad.cols());
      }
      vMax = vMax.array().max(vHat.array()).matrix();
      vHat = vMax;
    }

    // Update parameters: param = param - lr * mHat / (sqrt(vHat) + epsilon)
    param.array() -=
        static_cast<val_t>(learningRate_) * mHat.array() / (vHat.array().sqrt() + static_cast<val_t>(epsilon_));
  }

  void update(vec_t& param, const vec_t& grad, const std::string& paramId) override {
    // Increment parameter-specific timestep
    size_t& t = timestepVec_[paramId];
    t++;

    // Get or create first moment (mean) buffer
    auto& m = momentVec_[paramId];
    if (m.size() != grad.size()) {
      m = vec_t::Zero(grad.size());
    }

    // Get or create second moment (variance) buffer
    auto& v = varianceVec_[paramId];
    if (v.size() != grad.size()) {
      v = vec_t::Zero(grad.size());
    }

    // Apply weight decay if specified
    vec_t effectiveGrad = grad;
    if (weightDecay_ > 0.0) {
      effectiveGrad += static_cast<val_t>(weightDecay_) * param;
    }

    // Update biased first moment estimate
    m = static_cast<val_t>(beta1_) * m + static_cast<val_t>(1.0 - beta1_) * effectiveGrad;

    // Update biased second raw moment estimate
    v = static_cast<val_t>(beta2_) * v + static_cast<val_t>(1.0 - beta2_) * effectiveGrad.array().square().matrix();

    // Compute bias-corrected moments
    const val_t bias1Correction = static_cast<val_t>(1.0 / (1.0 - std::pow(beta1_, t)));
    vec_t mHat = bias1Correction * m;

    const val_t bias2Correction = static_cast<val_t>(1.0 / (1.0 - std::pow(beta2_, t)));
    vec_t vHat = bias2Correction * v;

    // AMSGrad variant
    if (amsgrad_) {
      auto& vMax = vMaxVec_[paramId];
      if (vMax.size() != grad.size()) {
        vMax = vec_t::Zero(grad.size());
      }
      vMax = vMax.array().max(vHat.array()).matrix();
      vHat = vMax;
    }

    // Update parameters
    param.array() -=
        static_cast<val_t>(learningRate_) * mHat.array() / (vHat.array().sqrt() + static_cast<val_t>(epsilon_));
  }

  void reset() override {
    momentMat_.clear();
    varianceMat_.clear();
    momentVec_.clear();
    varianceVec_.clear();
    vMaxMat_.clear();
    vMaxVec_.clear();
    timestepMat_.clear();
    timestepVec_.clear();
  }

  // Getters and setters
  double getBeta1() const {
    return beta1_;
  }
  double getBeta2() const {
    return beta2_;
  }
  double getEpsilon() const {
    return epsilon_;
  }
  double getWeightDecay() const {
    return weightDecay_;
  }
  bool isAMSGrad() const {
    return amsgrad_;
  }

  void setBeta1(double beta1) {
    beta1_ = beta1;
  }
  void setBeta2(double beta2) {
    beta2_ = beta2;
  }
  void setEpsilon(double epsilon) {
    epsilon_ = epsilon;
  }
  void setWeightDecay(double decay) {
    weightDecay_ = decay;
  }
  void setAMSGrad(bool amsgrad) {
    amsgrad_ = amsgrad;
  }

 private:
  double beta1_;        // Exponential decay rate for first moment
  double beta2_;        // Exponential decay rate for second moment
  double epsilon_;      // Small constant for numerical stability
  double weightDecay_;  // L2 regularization strength
  bool amsgrad_;        // Use AMSGrad variant

  // First moment (mean) estimates
  std::unordered_map<std::string, mat_t> momentMat_;
  std::unordered_map<std::string, vec_t> momentVec_;

  // Second moment (variance) estimates
  std::unordered_map<std::string, mat_t> varianceMat_;
  std::unordered_map<std::string, vec_t> varianceVec_;

  // Max of second moments (for AMSGrad)
  std::unordered_map<std::string, mat_t> vMaxMat_;
  std::unordered_map<std::string, vec_t> vMaxVec_;

  // Per-parameter timesteps
  std::unordered_map<std::string, size_t> timestepMat_;
  std::unordered_map<std::string, size_t> timestepVec_;
};

// Cyclical Learning Rate Optimizer
// Implements the policy from "Cyclical Learning Rates for Training Neural Networks"
// by Leslie N. Smith (https://arxiv.org/abs/1506.01186)
class CyclicLR : public Optimizer {
 public:
  enum class Policy {
    TRIANGULAR,   // Basic triangular cycle
    TRIANGULAR2,  // Triangular with amplitude halving each cycle
    EXP_RANGE     // Exponential amplitude decay
  };

  explicit CyclicLR(double baseLR = 0.001, double maxLR = 0.006, size_t stepSize = 2000,
      Policy policy = Policy::TRIANGULAR, double gamma = 0.99994, double momentum = 0.9,
      bool nesterov = false, double weightDecay = 0.0)
      : Optimizer(baseLR),
        baseLR_(baseLR),
        maxLR_(maxLR),
        stepSize_(stepSize),
        policy_(policy),
        gamma_(gamma),
        momentum_(momentum),
        nesterov_(nesterov),
        weightDecay_(weightDecay),
        globalStep_(0),
        cycleCount_(0) {
    if (baseLR_ >= maxLR_) {
      throw std::runtime_error("CyclicLR: base_lr must be less than max_lr");
    }
    if (stepSize_ == 0) {
      throw std::runtime_error("CyclicLR: step_size must be greater than 0");
    }
    currentLR_ = baseLR_;
  }

  virtual ~CyclicLR() = default;

  void update(mat_t& param, const mat_t& grad, const std::string& paramId) override {
    // Get or create velocity buffer
    auto& v = velocityMat_[paramId];
    if (v.rows() != grad.rows() || v.cols() != grad.cols()) {
      v = mat_t::Zero(grad.rows(), grad.cols());
    }

    // Apply weight decay if specified
    mat_t effectiveGrad = grad;
    if (weightDecay_ > 0.0) {
      effectiveGrad += static_cast<val_t>(weightDecay_) * param;
    }

    // Update velocity: v = momentum * v + grad
    v = static_cast<val_t>(momentum_) * v + effectiveGrad;

    // Update parameters
    if (nesterov_) {
      // Nesterov momentum
      param.noalias() -= static_cast<val_t>(currentLR_) * (static_cast<val_t>(momentum_) * v + effectiveGrad);
    } else {
      // Standard momentum
      param.noalias() -= static_cast<val_t>(currentLR_) * v;
    }
  }

  void update(vec_t& param, const vec_t& grad, const std::string& paramId) override {
    // Get or create velocity buffer
    auto& v = velocityVec_[paramId];
    if (v.size() != grad.size()) {
      v = vec_t::Zero(grad.size());
    }

    // Apply weight decay if specified
    vec_t effectiveGrad = grad;
    if (weightDecay_ > 0.0) {
      effectiveGrad += static_cast<val_t>(weightDecay_) * param;
    }

    // Update velocity
    v = static_cast<val_t>(momentum_) * v + effectiveGrad;

    // Update parameters
    if (nesterov_) {
      // Nesterov momentum
      param.noalias() -= static_cast<val_t>(currentLR_) * (static_cast<val_t>(momentum_) * v + effectiveGrad);
    } else {
      // Standard momentum
      param.noalias() -= static_cast<val_t>(currentLR_) * v;
    }
  }

  // Called after all parameters have been updated for one batch/step
  void step() override {
    globalStep_++;
    _updateLearningRate();
  }

  void reset() override {
    velocityMat_.clear();
    velocityVec_.clear();
    globalStep_ = 0;
    cycleCount_ = 0;
    currentLR_ = baseLR_;
  }

  double getLearningRate() const override {
    return getCurrentLearningRate();
  }

  // Getters
  double getCurrentLearningRate() const {
    return currentLR_;
  }
  double getBaseLR() const {
    return baseLR_;
  }
  double getMaxLR() const {
    return maxLR_;
  }
  size_t getStepSize() const {
    return stepSize_;
  }
  size_t getGlobalStep() const {
    return globalStep_;
  }
  size_t getCycleCount() const {
    return cycleCount_;
  }
  Policy getPolicy() const {
    return policy_;
  }

  // Setters
  void setBaseLR(double lr) {
    baseLR_ = lr;
  }
  void setMaxLR(double lr) {
    maxLR_ = lr;
  }
  void setStepSize(size_t size) {
    stepSize_ = size;
  }
  void setPolicy(Policy policy) {
    policy_ = policy;
  }
  void setGamma(double gamma) {
    gamma_ = gamma;
  }

 private:
  void _updateLearningRate() {
    // Calculate cycle position
    size_t cycle = globalStep_ / (2 * stepSize_);
    size_t stepInCycle = globalStep_ % (2 * stepSize_);

    // Update cycle count if we've started a new cycle
    if (cycle > cycleCount_) {
      cycleCount_ = cycle;
    }

    // Calculate position within current cycle (0 to 1)
    double x = 0.0;
    if (stepInCycle < stepSize_) {
      // Ascending phase
      x = static_cast<double>(stepInCycle) / static_cast<double>(stepSize_);
    } else {
      // Descending phase
      x = 1.0 - static_cast<double>(stepInCycle - stepSize_) / static_cast<double>(stepSize_);
    }

    // Calculate amplitude based on policy
    double amplitude = 1.0;
    switch (policy_) {
      case Policy::TRIANGULAR:
        // Constant amplitude
        amplitude = 1.0;
        break;

      case Policy::TRIANGULAR2:
        // Amplitude halves each cycle
        amplitude = 1.0 / std::pow(2.0, static_cast<double>(cycle));
        break;

      case Policy::EXP_RANGE:
        // Exponential decay of amplitude
        amplitude = std::pow(gamma_, static_cast<double>(globalStep_));
        break;
    }

    // Calculate current learning rate
    currentLR_ = baseLR_ + (maxLR_ - baseLR_) * x * amplitude;
  }

 private:
  double baseLR_;       // Minimum learning rate
  double maxLR_;        // Maximum learning rate
  size_t stepSize_;     // Half-cycle length in iterations
  Policy policy_;       // Cycle policy
  double gamma_;        // Decay rate for EXP_RANGE policy
  double momentum_;     // Momentum coefficient
  bool nesterov_;       // Use Nesterov momentum
  double weightDecay_;  // L2 regularization strength

  size_t globalStep_;  // Current global step counter
  size_t cycleCount_;  // Number of complete cycles
  double currentLR_;   // Current learning rate

  // Velocity buffers for momentum
  std::unordered_map<std::string, mat_t> velocityMat_;
  std::unordered_map<std::string, vec_t> velocityVec_;
};

// One Cycle Learning Rate Policy
// Implements "Super-Convergence" from "A disciplined approach to neural network hyper-parameters"
// by Leslie N. Smith (https://arxiv.org/abs/1803.09820)
class OneCycleLR : public Optimizer {
 public:
  explicit OneCycleLR(double maxLR = 0.1, size_t totalSteps = 10000, double pctStart = 0.3, double divFactor = 25.0,
      double finalDivFactor = 10000.0, double momentum = 0.9, double baseMomentum = 0.85,
      double maxMomentum = 0.95, bool nesterov = false, double weightDecay = 0.0)
      : Optimizer(maxLR / divFactor),
        maxLR_(maxLR),
        totalSteps_(totalSteps),
        pctStart_(pctStart),
        divFactor_(divFactor),
        finalDivFactor_(finalDivFactor),
        momentum_(momentum),
        baseMomentum_(baseMomentum),
        maxMomentum_(maxMomentum),
        nesterov_(nesterov),
        weightDecay_(weightDecay),
        globalStep_(0) {
    initialLR_ = maxLR_ / divFactor_;
    minLR_ = initialLR_ / finalDivFactor_;
    stepSizeUp_ = static_cast<size_t>(totalSteps_ * pctStart_);
    stepSizeDown_ = totalSteps_ - stepSizeUp_;
    currentLR_ = initialLR_;
    currentMomentum_ = maxMomentum_;
  }

  virtual ~OneCycleLR() = default;

  void update(mat_t& param, const mat_t& grad, const std::string& paramId) override {
    // Get or create velocity buffer
    auto& v = velocityMat_[paramId];
    if (v.rows() != grad.rows() || v.cols() != grad.cols()) {
      v = mat_t::Zero(grad.rows(), grad.cols());
    }

    // Apply weight decay
    mat_t effectiveGrad = grad;
    if (weightDecay_ > 0.0) {
      effectiveGrad += static_cast<val_t>(weightDecay_) * param;
    }

    // Update velocity with current momentum
    v = static_cast<val_t>(currentMomentum_) * v + effectiveGrad;

    // Update parameters
    if (nesterov_) {
      param.noalias() -= static_cast<val_t>(currentLR_) * (static_cast<val_t>(currentMomentum_) * v + effectiveGrad);
    } else {
      param.noalias() -= static_cast<val_t>(currentLR_) * v;
    }
  }

  void update(vec_t& param, const vec_t& grad, const std::string& paramId) override {
    // Get or create velocity buffer
    auto& v = velocityVec_[paramId];
    if (v.size() != grad.size()) {
      v = vec_t::Zero(grad.size());
    }

    // Apply weight decay
    vec_t effectiveGrad = grad;
    if (weightDecay_ > 0.0) {
      effectiveGrad += static_cast<val_t>(weightDecay_) * param;
    }

    // Update velocity with current momentum
    v = static_cast<val_t>(currentMomentum_) * v + effectiveGrad;

    // Update parameters
    if (nesterov_) {
      param.noalias() -= static_cast<val_t>(currentLR_) * (static_cast<val_t>(currentMomentum_) * v + effectiveGrad);
    } else {
      param.noalias() -= static_cast<val_t>(currentLR_) * v;
    }
  }

  // Called after all parameters have been updated for one batch/step
  void step() override {
    globalStep_++;
    _updateSchedule();
  }

  void reset() override {
    velocityMat_.clear();
    velocityVec_.clear();
    globalStep_ = 0;
    currentLR_ = initialLR_;
    currentMomentum_ = maxMomentum_;
  }

  double getLearningRate() const override {
    return getCurrentLearningRate();
  }

  // Getters
  double getCurrentLearningRate() const {
    return currentLR_;
  }
  double getCurrentMomentum() const {
    return currentMomentum_;
  }
  size_t getGlobalStep() const {
    return globalStep_;
  }

 private:
  void _updateSchedule() {
    if (globalStep_ < stepSizeUp_) {
      // Phase 1: Increase LR from initial to max, decrease momentum from max to base
      double pct = static_cast<double>(globalStep_) / static_cast<double>(stepSizeUp_);
      currentLR_ = initialLR_ + (maxLR_ - initialLR_) * pct;
      currentMomentum_ = maxMomentum_ - (maxMomentum_ - baseMomentum_) * pct;
    } else if (globalStep_ < totalSteps_) {
      // Phase 2: Decrease LR from max to min, increase momentum from base to max
      double pct = static_cast<double>(globalStep_ - stepSizeUp_) / static_cast<double>(stepSizeDown_);
      currentLR_ = maxLR_ - (maxLR_ - minLR_) * pct;
      currentMomentum_ = baseMomentum_ + (maxMomentum_ - baseMomentum_) * pct;
    } else {
      // After total steps, keep min LR and max momentum
      currentLR_ = minLR_;
      currentMomentum_ = maxMomentum_;
    }
  }

 private:
  double maxLR_;           // Maximum learning rate
  size_t totalSteps_;      // Total training steps
  double pctStart_;        // Percentage of cycle spent increasing LR
  double divFactor_;       // Initial LR = maxLR / divFactor
  double finalDivFactor_;  // Final LR = initialLR / finalDivFactor
  double momentum_;        // Default momentum (unused in OneCycle)
  double baseMomentum_;    // Minimum momentum
  double maxMomentum_;     // Maximum momentum
  bool nesterov_;          // Use Nesterov momentum
  double weightDecay_;     // L2 regularization

  double initialLR_;        // Starting learning rate
  double minLR_;            // Minimum learning rate
  size_t stepSizeUp_;       // Steps in phase 1
  size_t stepSizeDown_;     // Steps in phase 2
  size_t globalStep_;       // Current step
  double currentLR_;        // Current learning rate
  double currentMomentum_;  // Current momentum

  std::unordered_map<std::string, mat_t> velocityMat_;
  std::unordered_map<std::string, vec_t> velocityVec_;
};