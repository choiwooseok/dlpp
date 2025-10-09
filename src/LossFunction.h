#pragma once

#include <cmath>
#include <stdexcept>

#include "ThreadPool.h"
#include "types.h"

class MeanSquaredError {
 public:
  static val_t f(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("MSE: size mismatch");
    }

    const size_t size = label.size();
    const val_t* labelData = label.data();
    const val_t* predData = predicted.data();

    val_t loss = ParallelUtil::p_transform_reduce(labelData, predData, size, [](val_t l, val_t p) {
      val_t diff = l - p;
      return diff * diff;
    });

    return loss / static_cast<val_t>(size);
  }

  static Tensor df(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("MSE: size mismatch");
    }

    const size_t size = label.size();
    const val_t factor = val_t(2) / static_cast<val_t>(size);

    Tensor grad(predicted.shape());

    ParallelUtil::p_transform(predicted.data(), label.data(), grad.data(), size,
        [factor](val_t p, val_t l) { return factor * (p - l); });

    return grad;
  }
};

class RootMeanSquaredError {
 public:
  static val_t f(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("RMSE: size mismatch");
    }

    const size_t size = label.size();
    const val_t* labelData = label.data();
    const val_t* predData = predicted.data();

    val_t mse = ParallelUtil::p_transform_reduce(labelData, predData, size, [](val_t l, val_t p) {
      val_t diff = l - p;
      return diff * diff;
    });

    mse /= static_cast<val_t>(size);
    return std::sqrt(mse);
  }

  static Tensor df(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("RMSE: size mismatch");
    }

    val_t rmse = f(label, predicted);

    // Avoid division by zero
    if (rmse < 1e-8) {
      return Tensor(predicted.shape());
    }

    const size_t size = label.size();
    const val_t factor = val_t(1) / (static_cast<val_t>(size) * rmse);

    Tensor grad(predicted.shape());

    ParallelUtil::p_transform(predicted.data(), label.data(), grad.data(), size,
        [factor](val_t p, val_t l) { return factor * (p - l); });

    return grad;
  }
};

class MeanAbsoluteError {
 public:
  static val_t f(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("MAE: size mismatch");
    }

    const size_t size = label.size();
    const val_t* labelData = label.data();
    const val_t* predData = predicted.data();

    val_t loss =
        ParallelUtil::p_transform_reduce(labelData, predData, size, [](val_t l, val_t p) { return std::abs(l - p); });

    return loss / static_cast<val_t>(size);
  }

  static Tensor df(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("MAE: size mismatch");
    }

    const size_t size = label.size();
    const val_t factor = val_t(1) / static_cast<val_t>(size);

    Tensor grad(predicted.shape());

    ParallelUtil::p_transform(predicted.data(), label.data(), grad.data(), size, [factor](val_t p, val_t l) {
      val_t diff = p - l;
      if (diff > val_t(0)) {
        return factor;
      }
      if (diff < val_t(0)) {
        return -factor;
      }
      return val_t(0);  // Subgradient at 0
    });

    return grad;
  }
};

// Binary Cross Entropy (for binary classification)
// Expects predictions in range [0, 1] (use Sigmoid activation)
class BinaryCrossEntropy {
 private:
  static constexpr val_t EPSILON = 1e-7;

 public:
  static val_t f(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("BCE: size mismatch");
    }

    const size_t size = label.size();
    const val_t* labelData = label.data();
    const val_t* predData = predicted.data();

    val_t loss = ParallelUtil::p_transform_reduce(labelData, predData, size, [](val_t y, val_t p_raw) {
      val_t p = std::clamp(p_raw, EPSILON, val_t(1) - EPSILON);
      return -(y * std::log(p) + (val_t(1) - y) * std::log(val_t(1) - p));
    });

    return loss / static_cast<val_t>(size);
  }

  static Tensor df(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("BCE: size mismatch");
    }

    const size_t size = label.size();
    const val_t factor = val_t(1) / static_cast<val_t>(size);

    Tensor grad(predicted.shape());

    ParallelUtil::p_transform(predicted.data(), label.data(), grad.data(), size, [factor](val_t p_raw, val_t y) {
      val_t p = std::clamp(p_raw, EPSILON, val_t(1) - EPSILON);
      return -factor * (y / p - (val_t(1) - y) / (val_t(1) - p));
    });

    return grad;
  }
};

// Categorical Cross Entropy (for multi-class classification)
// Expects predictions as probability distribution (use Softmax activation)
// Labels should be one-hot encoded
class CategoricalCrossEntropy {
 private:
  static constexpr val_t EPSILON = 1e-7;

 public:
  static val_t f(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("CCE: size mismatch");
    }

    const size_t batchSize = label.shape(0);
    const size_t totalSize = label.size();
    const val_t* labelData = label.data();
    const val_t* predData = predicted.data();

    val_t totalLoss = ParallelUtil::p_transform_reduce(labelData, predData, totalSize, [](val_t y, val_t p_raw) {
      val_t p = std::clamp(p_raw, EPSILON, val_t(1) - EPSILON);
      return y * std::log(p);
    });

    return -totalLoss / static_cast<val_t>(batchSize);
  }

  static Tensor df(const Tensor& label, const Tensor& predicted) {
    if (label.size() != predicted.size()) {
      throw std::runtime_error("CCE: size mismatch");
    }

    const size_t batchSize = label.shape(0);
    const size_t totalSize = label.size();
    const val_t factor = val_t(1) / static_cast<val_t>(batchSize);

    Tensor grad(predicted.shape());

    ParallelUtil::p_transform(predicted.data(), label.data(), grad.data(), totalSize, [factor](val_t p_raw, val_t y) {
      val_t p = std::clamp(p_raw, EPSILON, val_t(1) - EPSILON);
      return -factor * y / p;
    });

    return grad;
  }
};

// Aliases for common usage
using MSE = MeanSquaredError;
using RMSE = RootMeanSquaredError;
using MAE = MeanAbsoluteError;
using BCE = BinaryCrossEntropy;
using CCE = CategoricalCrossEntropy;