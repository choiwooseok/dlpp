#include <chrono>

#include "Network.h"
#include "helper/MNISTData.h"

long long getCurrentEpochMillis() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  MNISTData data("../resource/mnist/mnist_train.csv");
  tensor_t in = data.getImages();
  tensor_t out = data.getLabels();

  Network nn;
  nn.addLayer(new FullyConnectedLayer(28 * 28, 128));
  nn.addLayer(new ReLULayer(128));
  nn.addLayer(new FullyConnectedLayer(128, 10));
  nn.addLayer(new SigmoidLayer(10));

  // Network nn;
  // nn.load("mnist_model.json");

  nn.infos();
  nn.train<MSE>(in, out, 1000, 0.01, false);
  nn.save("mnist_model_" + std::to_string(getCurrentEpochMillis()) + ".json");

  return 0;
}
