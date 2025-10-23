#include <chrono>

#include "Network.h"

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

  int numSamples = 4;
  mat_t in_(numSamples, 2);
  mat_t label_(numSamples, 1);

  in_.row(0) << 0.0f, 0.0f;
  in_.row(1) << 0.0f, 1.0f;
  in_.row(2) << 1.0f, 0.0f;
  in_.row(3) << 1.0f, 1.0f;

  for (int i = 0; i < numSamples; i++) {
    label_.row(i) << (in_(i, 0) == in_(i, 1) ? 0.0f : 1.0f);
  }

  tensor_t in = fromMat(in_);
  tensor_t label = fromMat(label_);

  // ============================================================
  // Training Configuration
  // ============================================================

  const int epochs = 10000;
  const double learningRate = 0.01;

  nn.train<MSE>(in, label, epochs, learningRate);

  // ============================================================
  // Save Model
  // ============================================================

  std::string modelName = "xor_model_" + std::to_string(getCurrentTimeMillis()) + ".json";
  nn.save(modelName);

  std::cout << "\n"
            << std::string(60, '=') << std::endl;
  std::cout << "Training completed!" << std::endl;
  std::cout << "Model saved as: " << modelName << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  return 0;
}
