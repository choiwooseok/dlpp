#pragma once

#include "BaseLayer.h"

class BaseActivation : public BaseLayer {
public:
  explicit BaseActivation(const string &name) : BaseLayer(name) {}
  virtual ~BaseActivation() = default;

  vec_t forward(const vec_t &input) override {
    this->input = input;
    return f(input);
  }

  vec_t backward(const vec_t &dY, double eta) override {
    vec_t dX = dY.array() * df(input).array();
    return dX;
  }

  // activation function
  virtual vec_t f(const vec_t &input) = 0;

  // derivative of activation function
  virtual vec_t df(const vec_t &input) = 0;

private:
  // cache
  vec_t input;
};