#pragma once

#include "types.h"
#include "Optimizer.h"

#include <numeric>
#include <string>

using namespace std;
class BaseLayer {
 public:
  explicit BaseLayer(string name) : name_(name), layerId_(genLayerId()) {}
  virtual ~BaseLayer() = default;

 public:
  // (N,C,H,W)
  virtual tensor_t forward(const tensor_t &X) = 0;
  virtual tensor_t backward(const tensor_t &dY) = 0;
  virtual void updateParams(Optimizer *optimizer) = 0;
  virtual void info() = 0;

  string getName() { return name_; }

 private:
  static string genLayerId() {
    static size_t counter = 0;
    return "layer_" + std::to_string(counter++);
  }

 protected:
  string name_;
  string layerId_;
};
