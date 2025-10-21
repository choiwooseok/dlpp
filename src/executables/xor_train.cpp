#include <chrono>

#include "Network.h"

int main(int argc, char **argv) {
  Network nn;
  nn.addLayer(new FullyConnectedLayer(2, 4));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new FullyConnectedLayer(4, 1));
  nn.addLayer(new SigmoidLayer());

  tensor_t in(4, 2);
  in.row(0) << 0.f, 0.f;
  in.row(1) << 0.f, 1.f;
  in.row(2) << 1.f, 0.f;
  in.row(3) << 1.f, 1.f;

  tensor_t label(4, 1);
  label.row(0) << 0.f;
  label.row(1) << 1.f;
  label.row(2) << 1.f;
  label.row(3) << 0.f;

  nn.infos();
  nn.train<MSE>(in, label, 50000, 0.01);
  nn.save("xor_model_" + std::to_string(getCurrentTimeMillis()) + ".json");

  return 0;
}
