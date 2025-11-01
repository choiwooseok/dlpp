#include <gtest/gtest.h>

#include "Network.h"
#include "helper/Cifar10Data.h"
#include "layers/BatchNormLayer.h"
#include "layers/DropoutLayer.h"

#include <iostream>
#include <iomanip>

using namespace std;

class CIFAR10Fixture : public ::testing::Test {
 protected:
  Cifar10Data data = Cifar10Data("../resource/cifar10/");
  tensor_t in;
  tensor_t label;
  size_t N;

  int correct;
  vector<int> classCorrect;
  vector<int> classTotal;
  vector<vector<int>> confusionMatrix;

  void SetUp() override {
    data.readTestData();

    in = fromMat(toEigenMatrix(data.getPixels()));
    label = fromMat(toEigenMatrix(data.getLabels()));

    N = in.shape[0];

    // Data normalization: Scale to [0, 1]
    cout << "Normalizing pixel values to [0, 1]..." << endl;
    for (size_t i = 0; i < in.totalSize(); ++i) {
      in[i] = in[i] / 255.0f;
    }

    in.reshapeInPlace({
        N,
        Cifar10Data::NUM_CHANNELS,
        Cifar10Data::IMAGE_HEIGHT,
        Cifar10Data::IMAGE_WIDTH,
    });

    correct = 0;
    classCorrect = vector<int>(Cifar10Data::NUM_CLASSES, 0);
    classTotal = vector<int>(Cifar10Data::NUM_CLASSES, 0);
    confusionMatrix = vector<vector<int>>(Cifar10Data::NUM_CLASSES, vector<int>(Cifar10Data::NUM_CLASSES, 0));
  }

  void TearDown() override {}

  void printResult(int correct, int N, const vector<int>& classCorrect, const vector<int>& classTotal, const vector<vector<int>>& confusionMatrix) {
    cout << string(60, '=') << endl;
    cout << "Test Results" << endl;
    cout << string(60, '=') << endl;

    double overallAccuracy = static_cast<double>(correct) / static_cast<double>(N);
    cout << "Total Correct: " << correct << " / " << N << endl;
    cout << "Overall Accuracy: " << fixed << setprecision(2) << (overallAccuracy * 100.0) << "%" << endl;

    // Per-class accuracy
    cout << "\n"
         << string(60, '-') << endl;
    cout << "Per-Class Accuracy:" << endl;
    cout << string(60, '-') << endl;

    for (int c = 0; c < Cifar10Data::NUM_CLASSES; ++c) {
      double classAcc = classTotal[c] > 0
                            ? static_cast<double>(classCorrect[c]) / static_cast<double>(classTotal[c])
                            : 0.0;

      cout << setw(12) << left << data.classToString(static_cast<Cifar10Data::Class>(c)) << ": "
           << fixed << setprecision(2) << (classAcc * 100.0) << "% ("
           << classCorrect[c] << "/" << classTotal[c] << ")" << endl;
    }

    // Confusion Matrix
    cout << "\n"
         << string(60, '-') << endl;
    cout << "Confusion Matrix (rows: true, cols: predicted):" << endl;
    cout << string(60, '-') << endl;

    // Matrix
    for (int i = 0; i < Cifar10Data::NUM_CLASSES; ++i) {
      cout << setw(10) << left << data.classToString(static_cast<Cifar10Data::Class>(i)) << ":";
      for (int j = 0; j < Cifar10Data::NUM_CLASSES; ++j) {
        cout << setw(5) << right << confusionMatrix[i][j];
      }
      cout << endl;
    }

    cout << string(60, '=') << endl;
  }
};

TEST_F(CIFAR10Fixture, CIFAR10) {
  Network nn;
  nn.load("cifar10_model_1761975372948.json");
  nn.infos();

  for (size_t i = 0; i < N; ++i) {
    tensor_t in_i = in.nth(i);
    tensor_t label_i = label.nth(i);

    tensor_t out_i = nn.forward(in_i);

    int predicted = max_element_idx(out_i.flatten());
    int expected = max_element_idx(label_i.flatten());

    classTotal[expected]++;
    confusionMatrix[expected][predicted]++;

    if (predicted == expected) {
      correct++;
      classCorrect[expected]++;
    } else {
      // cout << "predicted: " << data.classToString(static_cast<Cifar10Data::Class>(predicted)) << " (" << predicted << ")" << endl;
      // data.printImage(i);
    }

    cout << "\rProgress: " << i + 1 << "/" << N << std::flush;
  }
  cout << endl;

  printResult(correct, N, classCorrect, classTotal, confusionMatrix);

  EXPECT_GT(static_cast<double>(correct) / static_cast<double>(N), 0.70);
}
