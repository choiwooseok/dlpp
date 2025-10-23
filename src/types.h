#pragma once

#include <cassert>
#include <chrono>
#include <initializer_list>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Dense>

using namespace std;

typedef float val_t;
typedef Eigen::VectorXf vec_t;
typedef Eigen::MatrixXf mat_t;

using RowMajorMatrix =
    Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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

// N-dim Tensor (row-major, initializer_list indexer)
struct TensorND {
  std::vector<size_t> shape;
  std::vector<size_t> strides;
  vec_t data;

  TensorND() = default;
  explicit TensorND(const std::vector<size_t> &shape_) : shape(shape_) {
    _calcStrides();
    data = vec_t((int)totalSize());
    if (totalSize() > 0)
      data.setZero();
  }

  size_t ndim() const { return shape.size(); }

  size_t totalSize() const {
    if (shape.empty())
      return 0;
    return std::accumulate(shape.begin(), shape.end(), size_t(1),
                           std::multiplies<size_t>());
  }

  // initializer_list accessor: t({n,c,h,w})
  val_t &operator()(std::initializer_list<size_t> idxs) {
    std::vector<size_t> v(idxs);
    return data[(int)_index(v)];
  }

  // initializer_list accessor: t({n,c,h,w})
  val_t operator()(std::initializer_list<size_t> idxs) const {
    std::vector<size_t> v(idxs);
    return data[(int)_index(v)];
  }

  // accessor: t({n,c,h,w})
  val_t &at(size_t n, size_t c, size_t h, size_t w) {
    std::vector<size_t> v{n, c, h, w};
    return data[(int)_index(v)];
  }

  // accessor: t({n,c,h,w})
  val_t at(size_t n, size_t c, size_t h, size_t w) const {
    std::vector<size_t> v{n, c, h, w};
    return data[(int)_index(v)];
  }

  // flat access
  val_t &operator[](size_t i) { return data[(int)i]; }
  val_t operator[](size_t i) const { return data[(int)i]; }

  TensorND nth(size_t n) const {
    assert(ndim() >= 1);
    size_t sliceSize = totalSize() / shape[0];

    TensorND T;
    T.shape = std::vector<size_t>(shape.begin() + 1, shape.end());
    T._calcStrides();
    T.data = vec_t(sliceSize);
    size_t offset = n * sliceSize;
    for (size_t i = 0; i < sliceSize; ++i) {
      T.data[(int)i] = data[(int)(offset + i)];
    }
    return T;
  }

  // 2D RowMajor mapping (rows x cols). rows*cols must == totalSize()
  Eigen::Map<const RowMajorMatrix> asMatrixConst(size_t rows,
                                                 size_t cols) const {
    assert(totalSize() != 0);
    assert(rows * cols == totalSize());
    return Eigen::Map<const RowMajorMatrix>(data.data(), (int)rows, (int)cols);
  }

  Eigen::Map<RowMajorMatrix> asMatrix(size_t rows, size_t cols) {
    assert(totalSize() != 0);
    assert(rows * cols == totalSize());
    return Eigen::Map<RowMajorMatrix>(data.data(), (int)rows, (int)cols);
  }

  static TensorND fromFlat(const vec_t &v, const std::vector<size_t> &shape_) {
    TensorND T(shape_);
    assert((size_t)v.size() == T.totalSize());
    T.data = v;
    return T;
  }

  static TensorND fromMat(const mat_t &M) {
    if (M.size() == 0)
      return TensorND();
    TensorND T({(size_t)M.rows(), (size_t)M.cols()});
    for (int r = 0; r < M.rows(); ++r) {
      for (int c = 0; c < M.cols(); ++c) {
        T.data[(size_t)r * (size_t)M.cols() + (size_t)c] = M(r, c);
      }
    }
    return T;
  }

  void fill(val_t value) {
    if (data.size() == 0)
      return;
    data.setConstant(value);
  }

  vec_t flatten() const {
    if (data.size() == 0)
      return vec_t();
    return data;
  }

  // im2col for single sample n (useful to accelerate conv by GEMM)
  // returns matrix of size (C*kH*kW) x (outH*outW)
  mat_t im2colSample(size_t n, size_t kH, size_t kW, size_t stride = 1,
                     size_t pad = 0) const {
    assert(ndim() == 4); // expects (N,C,H,W)
    size_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    (void)N;
    size_t outH = (H + 2 * pad - kH) / stride + 1;
    size_t outW = (W + 2 * pad - kW) / stride + 1;
    size_t patchSize = C * kH * kW;
    size_t cols = outH * outW;
    mat_t M((int)patchSize, (int)cols);
    M.setZero();

    for (size_t oh = 0; oh < outH; ++oh) {
      for (size_t ow = 0; ow < outW; ++ow) {
        int col = (int)(oh * outW + ow);
        int idx = 0;
        for (size_t c = 0; c < C; ++c) {
          for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
              int ih = (int)oh * (int)stride + (int)kh - (int)pad;
              int iw = (int)ow * (int)stride + (int)kw - (int)pad;
              val_t v = val_t(0);
              if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                v = at(n, c, (size_t)ih, (size_t)iw);
              }
              M(idx++, col) = v;
            }
          }
        }
      }
    }
    return M;
  }

private:
  size_t _index(const std::vector<size_t> &idxs) const {
    assert(idxs.size() == ndim());
    size_t off = 0;
    for (size_t i = 0; i < ndim(); ++i) {
      assert(idxs[i] < shape[i]);
      off += idxs[i] * strides[i];
    }
    return off;
  }

  void _calcStrides() {
    strides.assign(ndim(), 0);
    if (ndim() == 0)
      return;
    strides[ndim() - 1] = 1;
    for (int i = (int)ndim() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape[i + 1];
  }
};

// N-dim Tensor (row-major, initializer_list indexer)
using tensor_t = TensorND;
