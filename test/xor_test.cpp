#include <gtest/gtest.h>

#include "Network.h"

using namespace std;

void _log(const vec_t &in, const vec_t &out, float expected) {
  cout << "Input: [" << in(0) << ", " << in(1) << "], Predicted: " << out(0)
       << ", Expected: " << expected << endl;
}

TEST(XORTestSuite, XORTEST) {
  Network nn;
  nn.load("xor_model.json");
  nn.infos();

  mat_t in_(6, 2);
  in_.row(0) << 0.0f, 0.0f;
  in_.row(1) << 0.0f, 1.0f;
  in_.row(2) << 1.0f, 0.0f;
  in_.row(3) << 1.0f, 1.0f;
  in_.row(4) << 3.0f, 1.0f;
  in_.row(5) << 3.0f, 3.0f;

  mat_t expected(6, 1);
  expected.row(0) << 0.f;
  expected.row(1) << 1.f;
  expected.row(2) << 1.f;
  expected.row(3) << 0.f;
  expected.row(4) << 1.f;
  expected.row(5) << 0.f;

  // Convert to tensor
  tensor_t in = TensorND::fromMat(in_);

  for (int i = 0; i < 6; i++) {
    tensor_t in_i = in.nth(i);

    tensor_t pred = nn.forward(in_i);

    vec_t v = pred.flatten();

    _log(in_i.flatten(), v, expected(i));

    EXPECT_EQ((v(0) >= 0.5f ? 1 : 0), (int)expected(i));
  }
}