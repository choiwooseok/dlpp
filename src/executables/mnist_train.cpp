#include <chrono>

#include "Network.h"
#include "helper/MNISTLoader.h"

long long getCurrentEpochMillis() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  MNISTLoader loader;
  loader.load("../resource/mnist/mnist_train.csv");
  vector<vector<double>> in = loader.getImages();
  vector<vector<double>> out = loader.getLabels();

  Network nn;
  nn.addLayer(new FullyConnectedLayer(28 * 28, 128));
  nn.addLayer(new ReLULayer(128));
  nn.addLayer(new FullyConnectedLayer(128, 10));
  nn.addLayer(new SigmoidLayer(10));

  // Network nn;
  // nn.load("../resource/model/mnist_model.json");

  nn.train<MSE>(in, out, 512, 0.01, true);

  nn.save("../resource/model/mnist_model_" +
          std::to_string(getCurrentEpochMillis()) + ".json");

  return 0;
}
