#pragma once

#include "types.h"

#include <numeric>
#include <string>

using namespace std;
class BaseLayer {
protected:
  vec_t input;
  vec_t output;

  string name;
  int numInput;
  int numOutput;

public:
  BaseLayer(string name, int numInput, int numOutput)
      : name(name), numInput(numInput), numOutput(numOutput){};
  virtual ~BaseLayer() = default;

  virtual vec_t forward(const vec_t &X) = 0;
  virtual vec_t backward(const vec_t &dY, double eta) = 0;

  string getName() { return name; }
  int getNumInput() const { return numInput; }
  int getNumOutput() const { return numOutput; }
};
