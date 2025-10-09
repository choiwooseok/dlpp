#pragma once

#include "layers/BaseLayer.h"

#include <memory>
#include <vector>

using namespace std;

enum class LossFunction { MSE, BCE };

class MLP {
public:
  void addLayer(BaseLayer *layer);

  vector<double> forward(const vector<double> &input);
  void backward(const vector<double> &err, double learningRate);

  void train(const vector<vector<double>> &input,
             const vector<vector<double>> &target, int epochs,
             double learningRate, LossFunction lossFunction = LossFunction::MSE,
             bool consoleOut = false);

  void save(const string &filepath);
  void load(const string &filepath);

private:
  double _bce(const vector<double> &target, const vector<double> &output);
  vector<double> _bceDerivative(const vector<double> &target,
                                const vector<double> &output);

  double _mse(const vector<double> &target, const vector<double> &output);
  vector<double> _mseDerivative(const vector<double> &target,
                                const vector<double> &output);

private:
  vector<unique_ptr<BaseLayer>> layers;
};