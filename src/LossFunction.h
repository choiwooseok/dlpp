#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "types.h"

class MeanSquaredError {
 public:
  static val_t f(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    return f(l, p);
  }

  static tensor_t df(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    vec_t g = df(l, p);
    tensor_t out(predicted.shape);
    assert((size_t)g.size() == out.totalSize());

    for (size_t i = 0; i < out.totalSize(); ++i) {
      out.data[i] = g(static_cast<int>(i));
    }
    return out;
  }

  static val_t f(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    val_t loss = transform_reduce(
        label.begin(), label.end(),
        predicted.begin(),
        val_t(0),
        plus<val_t>(),
        [](const val_t &l, const val_t &r) {
          return (l - r) * (l - r);
        });

    return loss / static_cast<val_t>(label.size());
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("label.size() != predicted.size()");
    }

    vec_t grad(static_cast<int>(label.size()));
    val_t factor = val_t(2) / (val_t)label.size();

    transform(
        label.begin(), label.end(),
        predicted.begin(),
        grad.data(),
        [factor](const val_t &l, const val_t &r) {
          return factor * (r - l);
        });

    return grad;
  }
};

class RootMeanSquaredError {
 public:
  static val_t f(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    return f(l, p);
  }

  static tensor_t df(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    vec_t g = df(l, p);
    tensor_t out(predicted.shape);
    assert((size_t)g.size() == out.totalSize());

    for (size_t i = 0; i < out.totalSize(); ++i) {
      out.data[i] = g(static_cast<int>(i));
    }
    return out;
  }

  static val_t f(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("RMSE: label.size() != predicted.size()");
    }

    val_t mse = transform_reduce(
        label.begin(), label.end(),
        predicted.begin(),
        val_t(0),
        plus<val_t>(),
        [](const val_t &l, const val_t &r) {
          return (l - r) * (l - r);
        });

    mse /= static_cast<val_t>(label.size());
    return std::sqrt(mse);
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("RMSE: label.size() != predicted.size()");
    }

    // d(RMSE)/dy = (y - label) / (N * RMSE)
    val_t rmse = f(label, predicted);

    // Avoid division by zero
    if (rmse < 1e-8) {
      return vec_t::Zero(label.size());
    }

    vec_t grad(static_cast<int>(label.size()));
    val_t factor = val_t(1) / (static_cast<val_t>(label.size()) * rmse);

    transform(
        label.begin(), label.end(),
        predicted.begin(),
        grad.data(),
        [factor](const val_t &l, const val_t &r) {
          return factor * (r - l);
        });

    return grad;
  }
};

class MeanAbsoluteError {
 public:
  static val_t f(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    return f(l, p);
  }

  static tensor_t df(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    vec_t g = df(l, p);
    tensor_t out(predicted.shape);
    assert((size_t)g.size() == out.totalSize());

    for (size_t i = 0; i < out.totalSize(); ++i) {
      out.data[i] = g(static_cast<int>(i));
    }
    return out;
  }

  static val_t f(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("MAE: label.size() != predicted.size()");
    }

    val_t loss = transform_reduce(
        label.begin(), label.end(),
        predicted.begin(),
        val_t(0),
        plus<val_t>(),
        [](const val_t &l, const val_t &r) {
          return std::abs(l - r);
        });

    return loss / static_cast<val_t>(label.size());
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("MAE: label.size() != predicted.size()");
    }

    // d(MAE)/dy = sign(y - label) / N
    vec_t grad(static_cast<int>(label.size()));
    val_t factor = val_t(1) / static_cast<val_t>(label.size());

    transform(
        label.begin(), label.end(),
        predicted.begin(),
        grad.data(),
        [factor](const val_t &l, const val_t &r) {
          val_t diff = r - l;
          if (diff > val_t(0)) return factor;
          if (diff < val_t(0)) return -factor;
          return val_t(0);  // Subgradient at 0
        });

    return grad;
  }
};

// Binary Cross Entropy (for binary classification)
// Expects predictions in range [0, 1] (use Sigmoid activation)
class BinaryCrossEntropy {
 public:
  static val_t f(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    return f(l, p);
  }

  static tensor_t df(const tensor_t &label, const tensor_t &predicted) {
    vec_t l = label.flatten();
    vec_t p = predicted.flatten();
    vec_t g = df(l, p);
    tensor_t out(predicted.shape);
    assert((size_t)g.size() == out.totalSize());

    for (size_t i = 0; i < out.totalSize(); ++i) {
      out.data[i] = g(static_cast<int>(i));
    }
    return out;
  }

  static val_t f(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("BCE: label.size() != predicted.size()");
    }

    const val_t epsilon = 1e-7;  // For numerical stability
    val_t loss = val_t(0);

    for (int i = 0; i < label.size(); ++i) {
      val_t y = label[i];
      val_t p = std::max(epsilon, std::min(val_t(1) - epsilon, predicted[i]));

      // BCE = -[y * log(p) + (1-y) * log(1-p)]
      loss += -(y * std::log(p) + (val_t(1) - y) * std::log(val_t(1) - p));
    }

    return loss / static_cast<val_t>(label.size());
  }

  static vec_t df(const vec_t &label, const vec_t &predicted) {
    if (label.size() != predicted.size()) {
      throw runtime_error("BCE: label.size() != predicted.size()");
    }

    const val_t epsilon = 1e-7;
    vec_t grad(static_cast<int>(label.size()));

    for (int i = 0; i < label.size(); ++i) {
      val_t y = label[i];
      val_t p = std::max(epsilon, std::min(val_t(1) - epsilon, predicted[i]));

      // d(BCE)/dp = -(y/p - (1-y)/(1-p)) / N
      grad[i] = -(y / p - (val_t(1) - y) / (val_t(1) - p)) /
                static_cast<val_t>(label.size());
    }

    return grad;
  }
};

// Categorical Cross Entropy (for multi-class classification)
// Expects predictions as probability distribution (use Softmax activation)
// Labels should be one-hot encoded
class CategoricalCrossEntropy {
 public:
  static val_t f(const tensor_t &label, const tensor_t &predicted) {
    // Expected shape: (batch_size, num_classes)
    assert(label.shape.size() >= 2);
    assert(predicted.shape.size() >= 2);

    const size_t batchSize = label.shape[0];
    const size_t numClasses = label.shape[1];

    assert(predicted.shape[0] == batchSize);
    assert(predicted.shape[1] == numClasses);

    const val_t epsilon = 1e-7;
    val_t totalLoss = val_t(0);

    for (size_t n = 0; n < batchSize; ++n) {
      val_t sampleLoss = val_t(0);

      for (size_t c = 0; c < numClasses; ++c) {
        const size_t idx = n * numClasses + c;
        val_t y = label.data[idx];
        val_t p = std::max(epsilon, std::min(val_t(1) - epsilon,
                                             predicted.data[idx]));

        // CCE = -sum(y * log(p))
        sampleLoss += y * std::log(p);
      }

      totalLoss -= sampleLoss;
    }

    return totalLoss / static_cast<val_t>(batchSize);
  }

  static tensor_t df(const tensor_t &label, const tensor_t &predicted) {
    assert(label.shape.size() >= 2);
    assert(predicted.shape.size() >= 2);

    const size_t batchSize = label.shape[0];
    const size_t numClasses = label.shape[1];

    tensor_t grad(predicted.shape);
    const val_t epsilon = 1e-7;
    const val_t factor = val_t(1) / static_cast<val_t>(batchSize);

    for (size_t n = 0; n < batchSize; ++n) {
      for (size_t c = 0; c < numClasses; ++c) {
        const size_t idx = n * numClasses + c;
        val_t y = label.data[idx];
        val_t p = std::max(epsilon, std::min(val_t(1) - epsilon,
                                             predicted.data[idx]));

        // d(CCE)/dp = -y/p / batch_size
        grad.data[idx] = -factor * y / p;
      }
    }

    return grad;
  }

  // Optimized version for Softmax + CCE combination
  // Returns: predicted - label (much simpler gradient)
  static tensor_t dfWithSoftmax(const tensor_t &label, const tensor_t &predicted) {
    assert(label.shape == predicted.shape);

    tensor_t grad(predicted.shape);
    const size_t totalSize = grad.totalSize();
    const val_t factor = val_t(1) / static_cast<val_t>(label.shape[0]);

    for (size_t i = 0; i < totalSize; ++i) {
      grad.data[i] = factor * (predicted.data[i] - label.data[i]);
    }

    return grad;
  }
};

// Alias for common usage
using MSE = MeanSquaredError;
using RMSE = RootMeanSquaredError;
using MAE = MeanAbsoluteError;
using BCE = BinaryCrossEntropy;
using CCE = CategoricalCrossEntropy;