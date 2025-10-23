#include <gtest/gtest.h>

#include "Network.h"
#include "helper/MNISTData.h"

#include <iostream>

using namespace std;

int _max_element_idx(const vec_t &vec) {
  return max_element(vec.begin(), vec.end()) - vec.begin();
}

TEST(MNistTestSuite, MNISTTEST) {
  Network nn;
  nn.load("mnist_model_1761228884178.json");
  nn.infos();

  MNISTData data("../resource/mnist/mnist_test.csv");
  tensor_t in = TensorND::fromMat(data.getImages());
  auto label = data.getLabels();

  int correct = 0;

  for (size_t i = 0; i < in.shape[0]; ++i) {
    // shape {1, featSize}
    tensor_t in_i = in.nth(i);

    tensor_t pred = nn.forward(in_i);

    int predicted = _max_element_idx(pred.flatten());
    int expected = _max_element_idx(label.row(i));

    if (predicted == expected) {
      correct++;
    } else {
      cout << "predicted: " << predicted << ", expected: " << expected << endl;
      data.print(in_i.flatten().transpose(), label.row(i));
    }
  }

  cout << "result : " << correct << " / " << in.shape[0] << endl;
  double accuracy = (double)correct / (double)in.shape[0];
  cout << "accuracy: " << accuracy << endl;

  EXPECT_GT(accuracy, 0.9);
}