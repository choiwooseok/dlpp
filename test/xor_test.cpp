#include <gtest/gtest.h>

#include "Network.h"

int _genRandomInt() {
  return abs(static_cast<int>(genRandom() * 1000)) % 10;
}

void _log(const vec_t& in, float out, float expected) {
  std::cout << std::format("Input: [{}, {}], prediected: {}, expected: {}", in(0), in(1), out, expected) << std::endl;
}

TEST(XORTestSuite, XORTest) {
  Network nn;
  nn.load("xor_model.json");

  int numSamples = 10000;
  mat_t in(numSamples, 2);
  mat_t label(numSamples, 1);

  for (int i = 0; i < numSamples; i++) {
    if (i % 5 == 0) {
      val_t v = _genRandomInt();
      in.row(i) << v, v;
    } else {
      in.row(i) << _genRandomInt(), _genRandomInt();
    }

    label.row(i) << (in(i, 0) == in(i, 1) ? 0.0f : 1.0f);
  }

  int correct = 0;
  for (int i = 0; i < numSamples; i++) {
    Tensor pred = nn.forward(Tensor::fromMat(in.row(i)));

    if ((pred[0] >= 0.5f ? 1 : 0) != label(i)) {
      _log(in.row(i), pred[0], label(i));
    } else {
      correct++;
    }
  }
  double accuracy = (double)correct / (double)numSamples;
  std::cout << std::format("Accuracy: {:.2f}% ({}/{})", accuracy * 100.0, correct, numSamples) << std::endl;

  EXPECT_GT(accuracy, 0.9);
}