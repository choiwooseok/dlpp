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
  nn.load("mnist_model.json");
  nn.infos();

  MNISTData data("../resource/mnist/mnist_test.csv");
  tensor_t in = data.getImages();
  tensor_t out = data.getLabels();

  int correct = 0;
  for (int i = 0; i < in.rows(); i++) {
    vec_t result = nn.forward(in.row(i));
    int label = _max_element_idx(out.row(i));
    int predicted = _max_element_idx(result);

    if (predicted == label) {
      correct++;
    } else {
      cout << "predicted: " << predicted << ", label: " << label << endl;
      data.print(in.row(i), out.row(i));
    }
  }

  cout << "result : " << correct << " / " << in.rows() << endl;
  double accuracy = (double)correct / in.rows();
  cout << "accuracy: " << accuracy << endl;

  EXPECT_GT(accuracy, 0.6);
}
