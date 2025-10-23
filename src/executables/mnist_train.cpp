#include <chrono>

#include "Network.h"
#include "helper/MNISTData.h"

int main(int argc, char **argv) {
  MNISTData data("../resource/mnist/mnist_train.csv");

  Network nn;
  nn.addLayer(new FullyConnectedLayer(28 * 28, 128));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new FullyConnectedLayer(128, 10));
  nn.addLayer(new SigmoidLayer());

  tensor_t in = TensorND::fromMat(data.getImages());
  tensor_t label = TensorND::fromMat(data.getLabels());

  nn.infos();
  nn.train<MSE>(in, label, 500, 0.01, 100, false);
  nn.save("mnist_model_" + std::to_string(getCurrentTimeMillis()) + ".json");

  return 0;
}
