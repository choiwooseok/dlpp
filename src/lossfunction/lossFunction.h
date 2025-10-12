#pragma once

#include <vector>

using namespace std;

enum class LossFunction { MSE, BCE };

class MSE {
public:
  static double f(const vector<double> &target, const vector<double> &output) {
    if (target.size() != output.size()) {
      throw runtime_error("target.size() != output.size()");
    }

    double sum = 0.0;
    for (int i = 0; i < target.size(); ++i) {
      double diff = target[i] - output[i];
      sum += diff * diff;
    }
    return sum / target.size();
  }

  static vector<double> df(const vector<double> &target,
                           const vector<double> &output) {
    if (target.size() != output.size()) {
      throw runtime_error("target.size() != output.size()");
    }
    vector<double> grad(target.size());
    double factor = 2.0 / target.size();
    for (int i = 0; i < target.size(); ++i) {
      grad[i] = factor * (output[i] - target[i]);
    }
    return grad;
  }
};

class BCE {
public:
  static double f(const vector<double> &target, const vector<double> &output) {
    if (target.size() != output.size()) {
      throw runtime_error("target.size() != output.size()");
    }

    double sum = 0.0;
    for (int i = 0; i < output.size(); i++) {
      sum += -target[i] * log(output[i]) -
             (1.0 - target[i]) * log((1.0 - output[i]));
    }
    return sum;
  }

  static vector<double> df(const vector<double> &target,
                           const vector<double> &output) {
    if (target.size() != output.size()) {
      throw runtime_error("target.size() != output.size()");
    }

    vector<double> grad(output.size());
    for (int i = 0; i < output.size(); i++) {
      grad[i] = (output[i] - target[i]) / (output[i] * (1.0 - output[i]));
    }
    return grad;
  }
};
