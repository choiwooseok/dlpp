#include "Network.h"
#include "helper/Cifar10Data.h"
#include "layers/BatchNormLayer.h"
#include "layers/DropoutLayer.h"

// Helper function for quick validation
void _quickValidation(Network& nn, const tensor_t& data, const tensor_t& labels, int numSamples) {
  int correct = 0;
  numSamples = std::min(numSamples, static_cast<int>(data.shape[0]));

  for (int i = 0; i < numSamples; ++i) {
    tensor_t sample = data.nth(i);
    tensor_t output = nn.forward(sample);

    vec_t out_vec = output.flatten();
    vec_t label_vec = labels.nth(i).flatten();

    int predicted = max_element_idx(out_vec);
    int expected = max_element_idx(label_vec);

    if (predicted == expected) {
      correct++;
    }

    cout << "\rProgress: " << i + 1 << "/" << numSamples << std::flush;
  }

  double accuracy = static_cast<double>(correct) / static_cast<double>(numSamples);
  cout << "Quick validation accuracy: " << (accuracy * 100.0) << "% (" << correct << "/" << numSamples << ")" << endl;
}

int main(int argc, char** argv) {
  // ============================================================
  // Network Preparation - CNN
  // ============================================================

  Network nn;
  // Input: 3x32x32

  // Conv Block 1: 32x32 -> 32x32
  nn.addLayer(new Conv2DLayer(3, 64, 3, 3, 1, 1, INIT::HE));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new BatchNormLayer(64));

  // Conv Block 2: 32x32 -> 32x32 -> 16x16
  nn.addLayer(new Conv2DLayer(64, 64, 3, 3, 1, 1, INIT::HE));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new BatchNormLayer(64));
  nn.addLayer(new MaxPoolingLayer(64, 2, 2, 2, 0));
  nn.addLayer(new DropoutLayer(0.25));

  // Conv Block 3: 16x16 -> 16x16
  nn.addLayer(new Conv2DLayer(64, 128, 3, 3, 1, 1, INIT::HE));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new BatchNormLayer(128));

  // Conv Block 4: 16x16 -> 16x16 -> 8x8
  nn.addLayer(new Conv2DLayer(128, 128, 3, 3, 1, 1, INIT::HE));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new BatchNormLayer(128));
  nn.addLayer(new MaxPoolingLayer(128, 2, 2, 2, 0));
  nn.addLayer(new DropoutLayer(0.25));

  nn.addLayer(new FlattenLayer());

  // Dense Block
  nn.addLayer(new FullyConnectedLayer(8 * 8 * 128, 512, INIT::HE));
  nn.addLayer(new ReLULayer());
  nn.addLayer(new BatchNormLayer(512));
  nn.addLayer(new DropoutLayer(0.5));

  // Output Layer
  nn.addLayer(new FullyConnectedLayer(512, 10, INIT::HE));
  nn.addLayer(new SoftmaxLayer());

  nn.infos();

  // ============================================================
  // Data Preparation
  // ============================================================

  Cifar10Data data("../resource/cifar10/");
  data.readAllTrainData();

  tensor_t in = fromMat(toEigenMatrix(data.getPixels()));
  tensor_t label = fromMat(toEigenMatrix(data.getLabels()));

  size_t N = in.shape[0];

  // Reshape: [N, 3072] → [N, 3, 32, 32]
  cout << "\nReshaping data..." << endl;
  in.reshapeInPlace({N,
                     Cifar10Data::NUM_CHANNELS,
                     Cifar10Data::IMAGE_HEIGHT,
                     Cifar10Data::IMAGE_WIDTH});

  // Data normalization: Scale to [0, 1]
  cout << "Normalizing pixel values to [0, 1]..." << endl;
  for (size_t i = 0; i < in.totalSize(); ++i) {
    in[i] = in[i] / 255.0f;
  }

  // ============================================================
  // Training Configuration
  // ============================================================

  const int epochs = 10;
  const int batchSize = 256;
  const int checkPoints = 0;

  Adam opt(1e-4, 0.9, 0.999);

  nn.train<CCE>(in, label, epochs, &opt, batchSize, checkPoints);

  // ============================================================
  // Save Model
  // ============================================================

  std::string modelName = "cifar10_model_" + std::to_string(getCurrentTimeMillis());
  std::string extension = ".json";
  nn.save(modelName + extension);

  // Quick validation on training set
  cout << "\nRunning quick validation on training set..." << endl;
  _quickValidation(nn, in, label, 1000);

  return 0;
}