#include "NetworkBuilder.h"
#include "helper/MNISTData.h"

int main(int argc, char** argv) {
  // ============================================================
  // Network Preparation
  // ============================================================

  Network nn = NetworkBuilder()
                   .conv(1, 32, 5, 5, 1, 2, INIT::HE)
                   .relu()
                   .pool("max", 32, 2, 2, 2, 0)
                   .conv(32, 16, 5, 5, 1, 0, INIT::HE)
                   .relu()
                   .pool("max", 16, 2, 2, 2, 0)
                   .conv(16, 128, 5, 5, 1, 0, INIT::HE)
                   .relu()
                   .flatten()
                   .fc(128, 64, INIT::HE)
                   .relu()
                   .fc(64, 10, INIT::HE)
                   .softmax()
                   .build();

  // ============================================================
  // Data Preparation
  // ============================================================

  MNISTData train = MNISTDataBuilder()
                        .withTrainData()
                        .build();

  Tensor train_data = Tensor::fromMat(toEigenMatrix(train.getImages()));
  Tensor train_label = Tensor::fromMat(toEigenMatrix(train.getLabels()));

  MNISTData test = MNISTDataBuilder()
                       .withTestData()
                       .build();

  Tensor test_data = Tensor::fromMat(toEigenMatrix(test.getImages()));
  Tensor test_label = Tensor::fromMat(toEigenMatrix(test.getLabels()));

  auto reshape = [](Tensor& tensor) {
    tensor.reshape({
        tensor.shape(0),
        MNISTData::NUM_CHANNELS,
        MNISTData::IMAGE_HEIGHT,
        MNISTData::IMAGE_WIDTH,
    });
  };

  reshape(train_data);
  reshape(test_data);

  // ============================================================
  // Training Configuration
  // ============================================================
  Adam opt;

  const int epochs = 30;
  const int batchSize = 64;

  double max_acc = 0.0;
  nn.train<CCE>(train_data, train_label, epochs, &opt, batchSize, [&]() {
    auto ret = nn.test<MNISTData>(test_data, test_label, [](int, int) {});
    if (ret.getAccuracy() > max_acc) {
      std::string fileName = std::format("mnist_model_cnn_{:.2f}.json", max_acc);
      const auto fullPath = ResourceManager::instance().getModelPath(fileName);
      std::filesystem::remove(fullPath);

      max_acc = ret.getAccuracy();
      nn.save(std::format("mnist_model_cnn_{:.2f}.json", max_acc));
    }
  });

  return 0;
}
