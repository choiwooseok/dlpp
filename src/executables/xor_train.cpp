#include <chrono>

#include "Network.h"

long long getCurrentEpochMillis() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  Network nn;
  nn.addLayer(new FullyConnectedLayer(2, 4));
  nn.addLayer(new ReLULayer(4));
  nn.addLayer(new FullyConnectedLayer(4, 1));
  nn.addLayer(new SigmoidLayer(1));

  vector<vector<double>> in = {
      {0.0, 0.0},
      {0.0, 1.0},
      {1.0, 0.0},
      {1.0, 1.0},
  };
  vector<vector<double>> out = {{0.0}, {1.0}, {1.0}, {0.0}};

  nn.train<MSE>(in, out, 50000, 0.01, true);
  nn.save("../resource/model/xor_model_" +
          std::to_string(getCurrentEpochMillis()) + ".json");

  return 0;
}
