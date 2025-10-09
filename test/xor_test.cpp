#include <gtest/gtest.h>

#include "MLP.h"

using namespace std;

void _log(const vector<double> &in, const vector<double> &out,
          double expected) {
  cout << "Input: [" << in[0] << ", " << in[1] << "], Predicted: " << out[0]
       << ", Expected: " << expected << endl;
}

TEST(XORTestSuite, XORTEST) {
  MLP nn;
  nn.load("../resource/model/xor_model.json");

  vector<vector<double>> in = {
      {0.0, 0.0},
      {0.0, 1.0},
      {1.0, 0.0},
      {1.0, 1.0},
  };

  vector<double> expected = {0.0, 1.0, 1.0, 0.0};
  double abs_err = 0.1;

  for (int i = 0; i < 4; i++) {
    vector<double> result = nn.forward(in[i]);
    _log(in[i], result, expected[i]);
    EXPECT_NEAR(result[0], expected[i], abs_err);
  }
}