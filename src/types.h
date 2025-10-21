#pragma once

#include <chrono>
#include <random>
#include <vector>

#include <Eigen/Eigen>

using namespace std;

typedef float val_t;
typedef Eigen::VectorXf vec_t;
typedef Eigen::MatrixXf tensor_t;

static val_t genRandom() {
  using namespace std::chrono;

  random_device rd;
  mt19937 gen(rd() ^ system_clock::now().time_since_epoch().count());
  uniform_real_distribution<> dis(val_t(-1), val_t(1));

  return dis(gen);
}

static long long
timePointToMillis(const std::chrono::steady_clock::time_point &tp) {
  using namespace std::chrono;
  return duration_cast<milliseconds>(tp.time_since_epoch()).count();
}

static long long getCurrentTimeMillis() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
}
