#include <chrono>

#include "Network.h"

int main(int argc, char **argv) {
  Network nn;
  nn.addLayer(new FullyConnectedLayer(2, 4));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new FullyConnectedLayer(4, 1));
  nn.addLayer(new SigmoidLayer());

  int numSamples = 4;
  mat_t in(numSamples, 2);
  mat_t label(numSamples, 1);

  in.row(0) << 0.0f, 0.0f;
  in.row(1) << 0.0f, 1.0f;
  in.row(2) << 1.0f, 0.0f;
  in.row(3) << 1.0f, 1.0f;

  label.row(0) << 0.0f;
  label.row(1) << 1.0f;
  label.row(2) << 1.0f;
  label.row(3) << 0.0f;

  nn.infos();
  nn.train<MSE>(TensorND::fromMat(in), TensorND::fromMat(label), 10000, 0.01);
  nn.save("xor_model_" + std::to_string(getCurrentTimeMillis()) + ".json");

  for (int i = 0; i < numSamples; i++) {
    tensor_t pred = nn.forward(TensorND::fromMat(in.row(i)));
    std::cout << "Input: [" << in(i, 0) << ", " << in(i, 1)
              << "], Predicted: " << (pred.at(0, 0, 0, 0) >= 0.5f ? 1 : 0)
              << "(" << pred.at(0, 0, 0, 0) << ")"
              << ", Expected: " << label(i) << std::endl;
  }

  return 0;
}
