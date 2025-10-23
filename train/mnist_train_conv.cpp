#include "Network.h"
#include "helper/MNISTData.h"

int main(int argc, char **argv) {
  MNISTData data("../resource/mnist/mnist_train.csv");

  Network nn;

  // Layer 1: Conv2D - 1채널 → 6채널, 28x28 유지
  nn.addLayer(new Conv2DLayer(1, 6, 5, 5, 1, 2)); // stride=1, pad=2
  nn.addLayer(new ReLULayer());

  // Layer 2: MaxPooling - 28x28 → 14x14
  // 6채널, 2x2 kernel, stride=2
  nn.addLayer(new MaxPoolingLayer(6, 2, 2, 2, 0));

  // Layer 3: Conv2D - 6채널 → 16채널, 14x14 → 10x10
  nn.addLayer(new Conv2DLayer(6, 16, 5, 5, 1, 0)); // stride=1, pad=0
  nn.addLayer(new ReLULayer());

  // Layer 4: MaxPooling - 10x10 → 5x5
  // 16채널, 2x2 kernel, stride=2
  nn.addLayer(new MaxPoolingLayer(16, 2, 2, 2, 0));

  // Layer 5: Flatten - (16, 5, 5) → 400
  nn.addLayer(new FlattenLayer());

  // Layer 6-8: Fully Connected Layers
  nn.addLayer(new FullyConnectedLayer(400, 120));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new FullyConnectedLayer(120, 84));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new FullyConnectedLayer(84, 10));
  nn.addLayer(new SigmoidLayer());

  tensor_t in = TensorND::fromMat(data.getImages());
  tensor_t label = TensorND::fromMat(data.getLabels());

  // [60000, 784] -> [60000, 1, 28, 28]
  in.reshapeInPlace({60000, 1, 28, 28});

  nn.infos();
  nn.train<MSE>(in, label, 1000, 0.01, 100, false);
  nn.save("mnist_cnn_model_" + std::to_string(getCurrentTimeMillis()) +
          ".json");

  return 0;
}
