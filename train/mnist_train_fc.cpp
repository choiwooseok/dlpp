#include "Network.h"
#include "helper/MNISTData.h"

int main(int argc, char **argv) {
  // ============================================================
  // Network Preparation
  // ============================================================

  Network nn;
  nn.addLayer(new FullyConnectedLayer(28 * 28, 128));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new FullyConnectedLayer(128, 10));
  nn.addLayer(new SoftmaxLayer());

  nn.infos();

  // ============================================================
  // Data Preparation
  // ============================================================

  MNISTData data("../resource/mnist/mnist_train.csv");

  // [60000, 784]
  tensor_t in = fromMat(toEigenMatrix(data.getImages()));
  tensor_t label = fromMat(toEigenMatrix(data.getLabels()));

  // ============================================================
  // Training Configuration
  // ============================================================

  const int epochs = 100;
  const double learningRate = 0.01;

  nn.train<MSE>(in, label, epochs, learningRate);

  // ============================================================
  // Save Model
  // ============================================================

  std::string modelName = "mnist_fc_model_" + std::to_string(getCurrentTimeMillis()) + ".json";
  nn.save(modelName);

  std::cout << "\n"
            << std::string(60, '=') << std::endl;
  std::cout << "Training completed!" << std::endl;
  std::cout << "Model saved as: " << modelName << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  return 0;
}
