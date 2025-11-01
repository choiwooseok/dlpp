#include "Network.h"
#include "helper/MNISTData.h"

int main(int argc, char **argv) {
  // ============================================================
  // Network Preparation
  // ============================================================

  Network nn;

  nn.addLayer(new Conv2DLayer(1, 6, 5, 5, 1, 2, INIT::HE));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new MaxPoolingLayer(6, 2, 2, 2, 0));

  nn.addLayer(new Conv2DLayer(6, 16, 5, 5, 1, 0, INIT::HE));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new MaxPoolingLayer(16, 2, 2, 2, 0));

  nn.addLayer(new Conv2DLayer(16, 120, 5, 5, 1, 0, INIT::HE));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new FlattenLayer());

  nn.addLayer(new FullyConnectedLayer(120, 84, INIT::HE));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new FullyConnectedLayer(84, 10, INIT::HE));
  nn.addLayer(new SoftmaxLayer());

  nn.infos();

  // ============================================================
  // Data Preparation
  // ============================================================

  MNISTData data("../resource/mnist/mnist_train.csv");

  tensor_t in = fromMat(toEigenMatrix(data.getImages()));
  tensor_t label = fromMat(toEigenMatrix(data.getLabels()));

  size_t numData = in.shape[0];

  // Reshape: [N, 784] → [N, 1, 28, 28]
  in.reshapeInPlace({numData,
                     MNISTData::NUM_CHANNELS,
                     MNISTData::IMAGE_HEIGHT,
                     MNISTData::IMAGE_WIDTH});

  // ============================================================
  // Training Configuration
  // ============================================================

  const int epochs = 30;
  GD opt(1e-4);

  nn.train<CCE>(in, label, epochs, &opt);

  // ============================================================
  // Save Model
  // ============================================================

  std::string modelName = "mnist_cnn_model_" + std::to_string(getCurrentTimeMillis());
  std::string extension = ".json";
  nn.save(modelName + extension);

  return 0;
}
