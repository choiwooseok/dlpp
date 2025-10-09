#include <chrono>

#include "NetworkBuilder.h"

int _genRandomInt() {
  return abs(static_cast<int>(genRandom() * 1000)) % 4;
}

int main(int argc, char **argv) {
  // ============================================================
  // Network Preparation
  // ============================================================

  Network nn = NetworkBuilder()
                   .fc(2, 4)
                   .relu()
                   .fc(4, 1)
                   .sigmoid()
                   .build();

  // ============================================================
  // Data Preparation
  // ============================================================

  int numSamples = 50000;
  mat_t in_(numSamples, 2);
  mat_t label_(numSamples, 1);

  for (int i = 0; i < numSamples; i++) {
    if (i % 5 == 0) {
      val_t v = _genRandomInt();
      in_.row(i) << v, v;
    } else {
      in_.row(i) << _genRandomInt(), _genRandomInt();
    }

    label_.row(i) << (in_(i, 0) == in_(i, 1) ? 0.0f : 1.0f);
  }

  Tensor in = Tensor::fromMat(in_);
  Tensor label = Tensor::fromMat(label_);

  // ============================================================
  // Training Configuration
  // ============================================================

  const int epochs = 30;
  GD opt(0.01);  // learning rate = 0.01

  nn.train<BCE>(in, label, epochs, &opt, 1, []() {});

  // ============================================================
  // Save Model
  // ============================================================

  nn.save(std::format("xor_model_{}.json", std::to_string(getCurrentTimeMillis())));

  return 0;
}
