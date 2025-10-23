#pragma once

#include "types.h"

#include <numeric>
#include <string>

using namespace std;
class BaseLayer {
 public:
  explicit BaseLayer(string name) : name(name) {}
  virtual ~BaseLayer() = default;

 public:
  // (N,C,H,W)
  virtual tensor_t forward(const tensor_t &X) = 0;
  virtual tensor_t backward(const tensor_t &dY) = 0;
  virtual void updateParams(double eta) = 0;
  virtual void info() = 0;

  string getName() { return name; }

 protected:
  string name;
};
