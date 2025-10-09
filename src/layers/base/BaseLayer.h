#pragma once

#include "Exceptions.h"
#include "Optimizer.h"

class BaseLayer {
 public:
  explicit BaseLayer(const std::string& name)
      : name_(name), layerId_(_genLayerId()) {}
  virtual ~BaseLayer() = default;

 public:
  // (N,C,H,W)
  virtual Tensor forward(const Tensor& X) = 0;
  virtual Tensor backward(const Tensor& dY) = 0;
  virtual void updateParams(Optimizer* optimizer) = 0;
  virtual void info() = 0;

  const std::string& getName() const {
    return name_;
  }

  virtual void setTraining(bool training) {}  // default no op

 private:
  static std::string _genLayerId() {
    static size_t counter = 0;
    return "layer_" + std::to_string(counter++);
  }

 protected:
  std::string name_;
  std::string layerId_;
};
