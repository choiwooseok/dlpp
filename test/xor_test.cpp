#include <gtest/gtest.h>

#include "Network.h"

using namespace std;

void _log(const Eigen::RowVectorXf &in, const vec_t &out, float expected) {
  cout << "Input: [" << in(0) << ", " << in(1) << "], Predicted: " << out(0)
       << ", Expected: " << expected << endl;
}

TEST(XORTestSuite, XORTEST) {
  Network nn;
  nn.load("xor_model.json");
  nn.infos();

  tensor_t in(6, 2);
  in.row(0) << 0.0f, 0.0f;
  in.row(1) << 0.0f, 1.0f;
  in.row(2) << 1.0f, 0.0f;
  in.row(3) << 1.0f, 1.0f;
  in.row(4) << 3.0f, 1.0f;
  in.row(5) << 3.0f, 3.0f;

  tensor_t expected(6, 1);
  expected.row(0) << 0.f;
  expected.row(1) << 1.f;
  expected.row(2) << 1.f;
  expected.row(3) << 0.f;
  expected.row(4) << 1.f;
  expected.row(5) << 0.f;

  for (int i = 0; i < 6; i++) {
    vec_t result = nn.forward(in.row(i));
    _log(in.row(i), result, expected(i));

    for (auto &r : result) {
      r = (r >= 0.9f ? 1 : 0);
    }

    EXPECT_EQ(result(0), expected(i));
  }
}