#include "NetworkBuilder.h"
#include "helper/MNISTData.h"

int main(int argc, char** argv) {
  // ============================================================
  // Network Preparation
  // ============================================================

  Network nn = NetworkBuilder()
                   .fc(28 * 28, 128)
                   .relu()
                   .fc(128, 10)
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

  // ============================================================
  // Training Configuration
  // ============================================================

  Adam opt;

  const int epochs = 30;
  const int batchSize = 16;

  double max_acc = 0.0;
  nn.train<CCE>(train_data, train_label, epochs, &opt, batchSize, [&]() {
    auto ret = nn.test<MNISTData>(test_data, test_label, [](int, int) {});
    if (ret.getAccuracy() > max_acc) {
      std::string fileName = std::format("mnist_model_fc_{:.2f}.json", max_acc);
      const auto fullPath = ResourceManager::instance().getModelPath(fileName);
      std::filesystem::remove(fullPath);

      max_acc = ret.getAccuracy();
      nn.save(std::format("mnist_model_fc_{:.2f}.json", max_acc));
    }
  });

  return 0;
}
