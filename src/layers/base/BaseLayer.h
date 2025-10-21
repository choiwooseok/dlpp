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
  virtual vec_t forward(const vec_t &X) = 0;
  virtual vec_t backward(const vec_t &dY, double eta) = 0;

  string getName() { return name; }

protected:
  string name;
};
