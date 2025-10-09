#include "MLP.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "layers/FullyConnectedLayer.h"
#include "layers/LReLULayer.h"
#include "layers/ReLULayer.h"
#include "layers/SigmoidLayer.h"

void MLP::addLayer(BaseLayer *layer) { layers.emplace_back(layer); }

vector<double> MLP::forward(const vector<double> &input) {
  vector<double> output = input;
  for (const auto &layer : layers) {
    output = layer->forward(output);
  }
  return output;
}

void MLP::backward(const vector<double> &err, double learning_rate) {
  vector<double> grad = err;
  for (int i = layers.size() - 1; i >= 0; --i) {
    grad = layers[i]->backward(grad, learning_rate);
  }
}

void MLP::train(const vector<vector<double>> &input,
                const vector<vector<double>> &target, int epochs,
                double learningRate, LossFunction lossFunction,
                bool consoleOut) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    double total_loss = 0.0;

    for (int i = 0; i < input.size(); ++i) {
      vector<double> output = forward(input[i]);
      vector<double> loss_derivative;

      switch (lossFunction) {
      case LossFunction::MSE:
        total_loss += _mse(target[i], output);
        loss_derivative = _mseDerivative(target[i], output);
        break;
      case LossFunction::BCE:
        total_loss += _bce(target[i], output);
        loss_derivative = _bceDerivative(target[i], output);
        break;
      }
      backward(loss_derivative, learningRate);
    }

    if (consoleOut) {
      if ((epoch + 1) % 10 == 0) {
        cout << "epoch " << epoch + 1 << "/" << epochs
             << ", loss: " << total_loss / input.size() << endl;
      }
    }
  }
}

double MLP::_bce(const vector<double> &target, const vector<double> &output) {
  double sum = 0.0;
  for (int i = 0; i < output.size(); i++) {
    sum += target[i] * log(output[i]) + (1 - target[i]) * log((1 - output[i]));
  }
  return -sum / target.size();
}

vector<double> MLP::_bceDerivative(const vector<double> &target,
                                   const vector<double> &output) {
  vector<double> grad(output.size());
  for (int i = 0; i < output.size(); i++) {
    grad[i] = (output[0] - target[0]) / (output[0] * (1 - output[0]));
  }

  return grad;
}

double MLP::_mse(const vector<double> &target, const vector<double> &output) {
  double sum = 0.0;
  for (int i = 0; i < target.size(); ++i) {
    double diff = target[i] - output[i];
    sum += diff * diff;
  }
  return sum / target.size();
}

vector<double> MLP::_mseDerivative(const vector<double> &target,
                                   const vector<double> &output) {
  vector<double> grad(target.size());
  for (int i = 0; i < target.size(); ++i) {
    grad[i] = 2.0f * (output[i] - target[i]);
  }

  return grad;
}

void MLP::save(const string &filepath) {
  using json = nlohmann::ordered_json;

  json model;

  for (const auto &layer : layers) {
    json layer_json;
    layer_json["type"] = layer->name;

    if (layer->name == "FullyConnected") {
      FullyConnectedLayer *layerPtr =
          static_cast<FullyConnectedLayer *>(layer.get());
      layer_json["numInput"] = layerPtr->getNumInput();
      layer_json["numOutput"] = layerPtr->getNumOutput();
      layer_json["weights"] = layerPtr->getWeights();
      layer_json["biases"] = layerPtr->getBiases();
    }

    model["layers"].push_back(layer_json);
  }
  ofstream file(filepath);
  file.clear();
  file << model.dump(2);
  file.close();
}

void MLP::load(const string &filepath) {
  using json = nlohmann::json;

  ifstream file(filepath);
  json model;
  file >> model;
  file.close();

  for (auto &layer_json : model["layers"]) {
    string type = layer_json["type"];
    if (type == "FullyConnected") {
      int numInput = layer_json["numInput"];
      int numOutput = layer_json["numOutput"];
      auto weights = layer_json["weights"].get<vector<vector<double>>>();
      auto biases = layer_json["biases"].get<vector<double>>();

      FullyConnectedLayer *fcLayer =
          new FullyConnectedLayer(numInput, numOutput);

      fcLayer->setWeights(weights);
      fcLayer->setBiases(biases);

      layers.emplace_back(fcLayer);

    } else if (type == "ReLU") {
      layers.emplace_back(new ReLULayer());
    } else if (type == "SigmoidLayer") {
      layers.emplace_back(new SigmoidLayer());
    } else if (type == "LReLU") {
      layers.emplace_back(new LReLULayer());
    } else {
      cerr << "Unknown layer type: " << type << endl;
    }
  }
}
