#include <chrono>

#include "Network.h"

long long getCurrentEpochMillis() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  Network nn;
  nn.addLayer(new FullyConnectedLayer(2, 4));
  nn.addLayer(new ReLULayer(4));
  nn.addLayer(new FullyConnectedLayer(4, 1));
  nn.addLayer(new SigmoidLayer(1));

  tensor_t in(4, 2);
  in.row(0) << 0.f, 0.f;
  in.row(1) << 0.f, 1.f;
  in.row(2) << 1.f, 0.f;
  in.row(3) << 1.f, 1.f;

  tensor_t out(4, 1);
  out.row(0) << 0.f;
  out.row(1) << 1.f;
  out.row(2) << 1.f;
  out.row(3) << 0.f;

  nn.infos();
  nn.train<MSE>(in, out, 50000, 0.01);
  nn.save("xor_model_" + std::to_string(getCurrentEpochMillis()) + ".json");

  return 0;
}
