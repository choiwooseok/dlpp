#include <gtest/gtest.h>

#include "Network.h"
#include "helper/MNISTData.h"

#include <iostream>

using namespace std;

int _max_element_idx(const vec_t &v) {
  return max_element(v.begin(), v.end()) - v.begin();
}

TEST(MNistTestSuite, M_FC) {
  Network nn;
  nn.load("mnist_fc_model.json");
  nn.infos();

  MNISTData data = MNISTData("../resource/mnist/mnist_test.csv");
  tensor_t in = fromMat(toEigenMatrix(data.getImages()));
  tensor_t label = fromMat(toEigenMatrix(data.getLabels()));

  int numTest = in.shape[0];
  int correct = 0;

  for (size_t i = 0; i < numTest; ++i) {
    tensor_t in_i = in.nth(i);
    tensor_t label_i = label.nth(i);

    tensor_t out_i = nn.forward(in_i);

    int predicted = _max_element_idx(out_i.flatten());
    int expected = _max_element_idx(label_i.flatten());

    if (predicted == expected) {
      correct++;
    } else {
      cout << "predicted: " << predicted << endl;
      data.print(i);
    }
  }

  cout << "result : " << correct << " / " << numTest << endl;
  double accuracy = static_cast<double>(correct) / static_cast<double>(numTest);
  cout << "accuracy: " << accuracy << endl;

  EXPECT_GT(accuracy, 0.9);
}

TEST(MNistTestSuite, M_CNN) {
  Network nn;
  nn.load("mnist_cnn_model.json");
  nn.infos();

  MNISTData data = MNISTData("../resource/mnist/mnist_test.csv");
  tensor_t in = fromMat(toEigenMatrix(data.getImages()));
  tensor_t label = fromMat(toEigenMatrix(data.getLabels()));

  int numTest = in.shape[0];
  int correct = 0;

  for (size_t i = 0; i < numTest; ++i) {
    tensor_t in_i = in.nth(i);
    tensor_t label_i = label.nth(i);

    in_i.reshapeInPlace({1, 1, 28, 28});
    tensor_t out_i = nn.forward(in_i);

    int predicted = _max_element_idx(out_i.flatten());
    int expected = _max_element_idx(label_i.flatten());

    if (predicted == expected) {
      correct++;
    } else {
      cout << "predicted: " << predicted << endl;
      data.print(i);
    }
  }

  cout << "result : " << correct << " / " << numTest << endl;
  double accuracy = static_cast<double>(correct) / static_cast<double>(numTest);
  cout << "accuracy: " << accuracy << endl;

  EXPECT_GT(accuracy, 0.9);
}