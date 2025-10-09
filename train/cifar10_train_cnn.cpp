#include "NetworkBuilder.h"
#include "helper/Cifar10Data.h"

int main(int argc, char** argv) {
  // ============================================================
  // Network Preparation
  // ============================================================
  Network nn = NetworkBuilder()
                   .conv(3, 32, 3, 3, 1, 1, INIT::HE)
                   .relu()
                   .pool("max", 32, 2, 2, 2, 0)
                   .conv(32, 64, 3, 3, 1, 1, INIT::HE)
                   .relu()
                   .pool("max", 64, 2, 2, 2, 0)
                   .flatten()
                   .fc(8 * 8 * 64, 64, INIT::HE)
                   .relu()
                   .fc(64, 10, INIT::HE)
                   .softmax()
                   .build();

  // ============================================================
  // Data Preparation
  // ============================================================

  Cifar10Data train_ds = Cifar10DataBuilder()
                             .withTrainData()
                             .build();

  Tensor train_data = Tensor::fromMat(toEigenMatrix(train_ds.getImages()));
  Tensor train_label = Tensor::fromMat(toEigenMatrix(train_ds.getLabels()));

  Cifar10Data test_ds = Cifar10DataBuilder()
                            .withTestData()
                            .build();

  Tensor test_data = Tensor::fromMat(toEigenMatrix(test_ds.getImages()));
  Tensor test_label = Tensor::fromMat(toEigenMatrix(test_ds.getLabels()));

  auto reshape = [](Tensor& tensor) {
    tensor.reshape({
        tensor.shape(0),
        Cifar10Data::NUM_CHANNELS,
        Cifar10Data::IMAGE_HEIGHT,
        Cifar10Data::IMAGE_WIDTH,
    });
  };

  reshape(train_data);
  reshape(test_data);

  // ============================================================
  // Training Configuration
  // ============================================================

  const int epochs = 100;
  const int batchSize = 128;

  Adam opt(1e-4);

  double max_acc = 0.0;
  nn.train<CCE>(train_data, train_label, epochs, &opt, batchSize, [&]() {
    auto ret = nn.test<Cifar10Data>(test_data, test_label, [&](int idx, int pred) {
      // std::cout << std::format("\npredicted: {} ({})\n",
      //     Cifar10Data::classToString(static_cast<Cifar10Data::Class>(pred)), pred);
    });
    if (ret.getAccuracy() > max_acc) {
      std::string fileName = std::format("cifar10_model_cnn_{:.2f}.json", max_acc);
      const auto fullPath = ResourceManager::instance().getModelPath(fileName);
      std::filesystem::remove(fullPath);

      max_acc = ret.getAccuracy();
      nn.save(std::format("cifar10_model_cnn_{:.2f}.json", max_acc));
    }
  });

  return 0;
}
