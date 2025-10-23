#include "Network.h"
#include "helper/MNISTData.h"

int main(int argc, char **argv) {
  // ============================================================
  // Network Preparation
  // ============================================================

  Network nn;

  // C1: Convolutional Layer
  // Input: 28x28x1 → Output: 28x28x6 (with padding to maintain size)
  // Original: 32x32x1 → 28x28x6, but MNIST is 28x28
  nn.addLayer(new Conv2DLayer(1, 6, 5, 5, 1, 2, INIT::HE));  // 5x5 kernel, 6 filters
  nn.addLayer(new ReLULayer());

  // S2: Subsampling (Pooling) Layer
  // Input: 28x28x6 → Output: 14x14x6
  nn.addLayer(new MaxPoolingLayer(6, 2, 2, 2, 0));  // 2x2 max pooling

  // C3: Convolutional Layer
  // Input: 14x14x6 → Output: 10x10x16
  nn.addLayer(new Conv2DLayer(6, 16, 5, 5, 1, 0, INIT::HE));  // 5x5 kernel, 16 filters
  nn.addLayer(new ReLULayer());

  // S4: Subsampling (Pooling) Layer
  // Input: 10x10x16 → Output: 5x5x16
  nn.addLayer(new MaxPoolingLayer(16, 2, 2, 2, 0));  // 2x2 max pooling

  // C5: Convolutional Layer (acts as Fully Connected)
  // Input: 5x5x16 → Output: 1x1x120 (essentially FC layer)
  nn.addLayer(new Conv2DLayer(16, 120, 5, 5, 1, 0, INIT::HE));  // 5x5 kernel, 120 filters
  nn.addLayer(new ReLULayer());

  // Flatten: → 120
  nn.addLayer(new FlattenLayer());

  // F6: Fully Connected Layer
  // Input: 120 → Output: 84
  nn.addLayer(new FullyConnectedLayer(120, 84, INIT::HE));
  nn.addLayer(new ReLULayer());

  // Output Layer: Fully Connected
  // Input: 84 → Output: 10
  nn.addLayer(new FullyConnectedLayer(84, 10, INIT::HE));
  nn.addLayer(new SoftmaxLayer());  // Use Softmax for classification

  nn.infos();

  // ============================================================
  // Data Preparation
  // ============================================================

  MNISTData data("../resource/mnist/mnist_train.csv");

  tensor_t in = fromMat(toEigenMatrix(data.getImages()));
  tensor_t label = fromMat(toEigenMatrix(data.getLabels()));

  // Reshape: [60000, 784] → [60000, 1, 28, 28]
  in.reshapeInPlace({60000, 1, 28, 28});

  // ============================================================
  // Training Configuration
  // ============================================================

  const int epochs = 30;
  const double learningRate = 0.001;

  nn.train<MSE>(in, label, epochs, learningRate);

  // ============================================================
  // Save Model
  // ============================================================

  std::string modelName = "mnist_cnn_model_" + std::to_string(getCurrentTimeMillis()) + ".json";
  nn.save(modelName);

  std::cout << "\n"
            << std::string(60, '=') << std::endl;
  std::cout << "Training completed!" << std::endl;
  std::cout << "Model saved as: " << modelName << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  return 0;
}
