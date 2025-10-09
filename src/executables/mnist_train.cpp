#include <chrono>

#include "MLP.h"
#include "helper/MNISTLoader.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/ReLULayer.h"
#include "layers/SigmoidLayer.h"

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

  // MLP nn;
  // nn.addLayer(new FullyConnectedLayer(28 * 28, 128));
  // nn.addLayer(new ReLULayer());
  // nn.addLayer(new FullyConnectedLayer(128, 10));
  // nn.addLayer(new SigmoidLayer());

  MLP nn;
  nn.load("../resource/model/mnist_model.json");

  nn.train(in, out, 512, 0.001, LossFunction::MSE, true);

  nn.save("../resource/model/mnist_model_" +
          std::to_string(getCurrentEpochMillis()) + ".json");

  return 0;
}
