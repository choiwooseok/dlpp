#include <chrono>

#include "Network.h"

int _genRandomInt() { return abs(static_cast<int>(genRandom() * 1000)) % 4; }

int main(int argc, char **argv) {
  // ============================================================
  // Network Preparation
  // ============================================================

  Network nn;
  nn.addLayer(new FullyConnectedLayer(2, 4));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new FullyConnectedLayer(4, 1));
  nn.addLayer(new SigmoidLayer());

  nn.infos();

  // ============================================================
  // Data Preparation
  // ============================================================

  int numSamples = 50000;
  mat_t in_(numSamples, 2);
  mat_t label_(numSamples, 1);

  for (int i = 0; i < numSamples; i++) {
    if (i % 5 == 0) {
      val_t v = _genRandomInt();
      in_.row(i) << v, v;
    } else {
      in_.row(i) << _genRandomInt(), _genRandomInt();
    }

    label_.row(i) << (in_(i, 0) == in_(i, 1) ? 0.0f : 1.0f);
  }

  tensor_t in = fromMat(in_);
  tensor_t label = fromMat(label_);

  // ============================================================
  // Training Configuration
  // ============================================================

  const int epochs = 30;
  GD opt(0.01);  // learning rate = 0.01

  nn.train<BCE>(in, label, epochs, &opt);

  // ============================================================
  // Save Model
  // ============================================================

  std::string modelName = "xor_model_" + std::to_string(getCurrentTimeMillis());
  std::string extension = ".json";
  nn.save(modelName + extension);

  return 0;
}
