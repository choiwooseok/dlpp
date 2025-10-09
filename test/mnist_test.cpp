#include <gtest/gtest.h>

#include "Network.h"
#include "helper/MNISTData.h"

class MNISTFixture : public ::testing::TestWithParam<std::string> {
 protected:
  Network nn;

  MNISTData data = MNISTDataBuilder()
                       .withTestData()
                       .build();
  Tensor test_data = Tensor::fromMat(toEigenMatrix(data.getImages()));
  Tensor test_label = Tensor::fromMat(toEigenMatrix(data.getLabels()));

  void SetUp() override {
    test_data.reshape({
        test_data.shape(0),
        MNISTData::NUM_CHANNELS,
        MNISTData::IMAGE_HEIGHT,
        MNISTData::IMAGE_WIDTH,
    });

    nn.load(GetParam());
  }

  void TearDown() override {}
};

TEST_P(MNISTFixture, MNISTTest) {
  auto ret = nn.test<MNISTData>(test_data, test_label, [&](int idx, int predicted) {
    std::cout << std::format("\npredicted: {} ({})\n",
        MNISTData::classToString(static_cast<MNISTData::Class>(predicted)),
        predicted);
    data.printData(idx);
  });

  EXPECT_GT(ret.getAccuracy(), 0.98);
}

INSTANTIATE_TEST_SUITE_P(
    MNIST, MNISTFixture, ::testing::Values("mnist_model_fc_0.98.json", "mnist_model_cnn_0.99.json"));
