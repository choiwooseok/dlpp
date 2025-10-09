#include <gtest/gtest.h>

#include "MLP.h"
#include "helper/MNISTLoader.h"

#include <iostream>

using namespace std;

TEST(MNistTestSuite, MNISTTEST) {
  MLP nn;
  nn.load("../resource/model/mnist_model.json");

  MNISTLoader loader;
  loader.load("../resource/mnist/mnist_test.csv");
  vector<vector<double>> in = loader.getImages();
  vector<vector<double>> out = loader.getLabels();

  int correct = 0;
  for (int i = 0; i < in.size(); i++) {
    vector<double> result = nn.forward(in[i]);
    int predicted = max_element(result.begin(), result.end()) - result.begin();
    int actual = max_element(out[i].begin(), out[i].end()) - out[i].begin();
    cout << "predicted: " << predicted << ", actual: " << actual << endl;

    if (predicted == actual) {
      correct++;
    } else {
      loader.printData(in[i], out[i]);
    }
  }
  double accuracy = (double)correct / in.size();
  cout << "accuracy: " << accuracy << endl;
  EXPECT_GT(accuracy, 0.6);
}
