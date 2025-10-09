#include <gtest/gtest.h>

#include "Network.h"
#include "helper/Cifar10Data.h"

class CIFAR10Fixture : public ::testing::TestWithParam<std::string> {
 protected:
  Network nn;

  Cifar10Data data = Cifar10DataBuilder()
                         .withTestData()
                         .build();
  Tensor test_data = Tensor::fromMat(toEigenMatrix(data.getImages()));
  Tensor test_label = Tensor::fromMat(toEigenMatrix(data.getLabels()));

  void SetUp() override {
    test_data.reshape({
        test_data.shape(0),
        Cifar10Data::NUM_CHANNELS,
        Cifar10Data::IMAGE_HEIGHT,
        Cifar10Data::IMAGE_WIDTH,
    });

    nn.load(GetParam());
  }

  void TearDown() override {}
};

TEST_P(CIFAR10Fixture, CIFAR10) {
  auto ret = nn.test<Cifar10Data>(test_data, test_label, [&](int idx, int predicted) {
    std::cout << std::format("\npredicted: {} ({})\n",
        Cifar10Data::classToString(static_cast<Cifar10Data::Class>(predicted)),
        predicted);
    data.printData(idx);
  });

  EXPECT_GT(ret.getAccuracy(), 0.67);
}

INSTANTIATE_TEST_SUITE_P(CIFAR10, CIFAR10Fixture, ::testing::Values("cifar10_model_cnn_0.67.json"));
