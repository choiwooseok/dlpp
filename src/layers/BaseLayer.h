#pragma once

#include <numeric>
#include <string>
#include <vector>

using namespace std;
class BaseLayer {
public:
  vector<double> input;
  vector<double> output;
  string name;

public:
  BaseLayer(string name) : name(name){};
  virtual ~BaseLayer() = default;

  virtual vector<double> forward(const vector<double> &input) = 0;
  virtual vector<double> backward(const vector<double> &err,
                                  double learningRate) = 0;

  string getName() { return name; }
};
