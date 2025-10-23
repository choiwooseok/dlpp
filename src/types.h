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

// N-dimensional Tensor with row-major layout
struct TensorND {
  std::vector<size_t> shape;
  std::vector<size_t> strides;
  vec_t data;

  // Constructors
  TensorND() = default;

  explicit TensorND(const std::vector<size_t> &shape_) : shape(shape_) {
    calculateStrides();
    allocateData();
  }

  // Dimension queries
  size_t ndim() const { return shape.size(); }

  void printShape() const {
    cout << "Tensor shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
      cout << shape[i];
      if (i != shape.size() - 1) {
        cout << ", ";
      }
    }
    cout << "]" << endl;
  }

  size_t totalSize() const {
    if (shape.empty()) {
      return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), size_t(1),
                           std::multiplies<size_t>());
  }

  bool isEmpty() const { return totalSize() == 0; }

  // Element access - initializer_list
  inline val_t &operator()(std::initializer_list<size_t> idxs) {
    return data[static_cast<int>(computeIndex(idxs))];
  }

  inline val_t operator()(std::initializer_list<size_t> idxs) const {
    return data[static_cast<int>(computeIndex(idxs))];
  }

  // Element access - explicit 4D
  inline val_t &at(size_t n, size_t c, size_t h, size_t w) {
    const size_t idx =
        n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
    return data[static_cast<int>(idx)];
  }

  inline val_t at(size_t n, size_t c, size_t h, size_t w) const {
    const size_t idx =
        n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
    return data[static_cast<int>(idx)];
  }

  // Flat access
  inline val_t &operator[](size_t i) {
    assert(i < totalSize());
    return data[static_cast<int>(i)];
  }

  inline val_t operator[](size_t i) const {
    assert(i < totalSize());
    return data[static_cast<int>(i)];
  }

  // Slice operations
  TensorND nth(size_t n) const {
    assert(ndim() >= 1);
    assert(n < shape[0]);

    return extractSlice(n);
  }

  // Matrix mapping
  Eigen::Map<const RowMajorMatrix> asMatrixConst(size_t rows,
                                                 size_t cols) const {
    validateMatrixDimensions(rows, cols);
    return Eigen::Map<const RowMajorMatrix>(data.data(), static_cast<int>(rows),
                                            static_cast<int>(cols));
  }

  Eigen::Map<RowMajorMatrix> asMatrix(size_t rows, size_t cols) {
    validateMatrixDimensions(rows, cols);
    return Eigen::Map<RowMajorMatrix>(data.data(), static_cast<int>(rows),
                                      static_cast<int>(cols));
  }

  // Factory methods
  static TensorND fromFlat(const vec_t &v, const std::vector<size_t> &shape_) {
    TensorND T(shape_);
    assert(static_cast<size_t>(v.size()) == T.totalSize());
    T.data = v;
    return T;
  }

  static TensorND fromMat(const mat_t &M) {
    if (M.size() == 0) {
      return TensorND();
    }

    TensorND T({static_cast<size_t>(M.rows()), static_cast<size_t>(M.cols())});
    copyMatrixData(M, T);
    return T;
  }

  // Data manipulation
  void fill(val_t value) {
    if (isEmpty()) {
      return;
    }
    data.setConstant(value);
  }

  vec_t flatten() const {
    if (isEmpty()) {
      return vec_t();
    }
    return data;
  }

  // Reshape tensor to new shape (must preserve total size)
  TensorND reshape(const std::vector<size_t> &newShape) const {
    // Validate that total size is preserved
    size_t newSize = 1;
    for (size_t dim : newShape) {
      newSize *= dim;
    }
    assert(newSize == totalSize() &&
           "Reshape: new shape must have same total size");

    // Create new tensor with new shape but same data
    TensorND result;
    result.shape = newShape;
    result.calculateStrides();
    result.data = data; // Share the same data

    return result;
  }

  // In-place reshape (modifies current tensor)
  void reshapeInPlace(const std::vector<size_t> &newShape) {
    // Validate that total size is preserved
    size_t newSize = 1;
    for (size_t dim : newShape) {
      newSize *= dim;
    }
    assert(newSize == totalSize() &&
           "Reshape: new shape must have same total size");

    // Update shape and recalculate strides
    shape = newShape;
    calculateStrides();
  }

  // Convenience: reshape with initializer list
  TensorND reshape(std::initializer_list<size_t> newShape) const {
    return reshape(std::vector<size_t>(newShape));
  }

  void reshapeInPlace(std::initializer_list<size_t> newShape) {
    reshapeInPlace(std::vector<size_t>(newShape));
  }

  // im2col transformation for convolution
  mat_t im2colSample(size_t sampleIdx, int kernelHeight, int kernelWidth,
                     int stride = 1, int pad = 0) const {
    assert(ndim() == 4); // Expected format: (N, C, H, W)

    const auto dims =
        extractConvDimensions(kernelHeight, kernelWidth, stride, pad);
    mat_t columns(dims.patchSize, dims.numColumns);

    // Optimized fill with direct pointer access
    fillColumnMatrix(columns, sampleIdx, dims, stride, pad);

    return columns;
  }

private:
  // Stride calculation
  void calculateStrides() {
    const size_t n = ndim();
    if (n == 0) {
      return;
    }

    strides.resize(n);
    strides[ndim() - 1] = 1;
    for (int i = static_cast<int>(ndim()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }

  // Memory allocation
  void allocateData() {
    const size_t size = totalSize();
    data = vec_t(static_cast<int>(size));
    if (size > 0) {
      data.setZero();
    }
  }

  // Index computation
  size_t computeIndex(std::initializer_list<size_t> idxs) const {
    std::vector<size_t> indices(idxs);
    return computeIndex(indices);
  }

  size_t computeIndex(const std::vector<size_t> &idxs) const {
    assert(idxs.size() == ndim());

    size_t offset = 0;
    for (size_t i = 0; i < ndim(); ++i) {
      assert(idxs[i] < shape[i]);
      offset += idxs[i] * strides[i];
    }
    return offset;
  }

  // Slice extraction
  TensorND extractSlice(size_t sliceIdx) const {
    const size_t sliceSize = totalSize() / shape[0];
    const size_t offset = sliceIdx * sliceSize;

    TensorND slice;
    slice.shape = std::vector<size_t>(shape.begin() + 1, shape.end());
    slice.calculateStrides();
    slice.data = vec_t(static_cast<int>(sliceSize));

    const val_t *srcPtr = data.data() + offset;
    std::memcpy(slice.data.data(), srcPtr, sliceSize * sizeof(val_t));

    return slice;
  }

  // Matrix operations
  void validateMatrixDimensions(size_t rows, size_t cols) const {
    assert(!isEmpty());
    assert(rows * cols == totalSize());
  }

  static void copyMatrixData(const mat_t &M, TensorND &T) {
    const int rows = M.rows();
    const int cols = M.cols();

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                           static_cast<size_t>(c);
        T.data[static_cast<int>(idx)] = M(r, c);
      }
    }
  }

  // im2col helper structures and methods
  struct ConvDimensions {
    int channels;
    int inputHeight;
    int inputWidth;
    int kernelHeight;
    int kernelWidth;
    int outputHeight;
    int outputWidth;
    int patchSize;
    int numColumns;
  };

  ConvDimensions extractConvDimensions(int kernelHeight, int kernelWidth,
                                       int stride, int pad) const {
    ConvDimensions dims;
    dims.channels = static_cast<int>(shape[1]);
    dims.inputHeight = static_cast<int>(shape[2]);
    dims.inputWidth = static_cast<int>(shape[3]);
    dims.kernelHeight = kernelHeight;
    dims.kernelWidth = kernelWidth;
    dims.outputHeight =
        calculateOutputDim(dims.inputHeight, kernelHeight, stride, pad);
    dims.outputWidth =
        calculateOutputDim(dims.inputWidth, kernelWidth, stride, pad);
    dims.patchSize = dims.channels * kernelHeight * kernelWidth;
    dims.numColumns = dims.outputHeight * dims.outputWidth;
    return dims;
  }

  static int calculateOutputDim(int inputDim, int kernelDim, int stride,
                                int pad) {
    return (inputDim + 2 * pad - kernelDim) / stride + 1;
  }

  mat_t createColumnMatrix(const ConvDimensions &dims) const {
    mat_t M(dims.patchSize, dims.numColumns);
    M.setZero();
    return M;
  }

  void fillColumnMatrix(mat_t &columns, size_t sampleIdx,
                        const ConvDimensions &dims, int stride, int pad) const {
    // Direct pointer access to input data
    const val_t *inData = data.data();
    const size_t sampleOffset = sampleIdx * strides[0];

    // Precompute channel strides
    const size_t channelStride = strides[1];
    const size_t rowStride = strides[2];

    int colIdx = 0;

    // Iterate over output spatial positions
    for (int oh = 0; oh < dims.outputHeight; ++oh) {
      for (int ow = 0; ow < dims.outputWidth; ++ow) {
        int patchIdx = 0;

        // Iterate over each channel
        for (int c = 0; c < dims.channels; ++c) {
          const size_t channelOffset = sampleOffset + c * channelStride;

          // Iterate over kernel spatial positions
          for (int kh = 0; kh < dims.kernelHeight; ++kh) {
            const int ih = oh * stride + kh - pad;

            // Early check for row validity
            if (ih >= 0 && ih < dims.inputHeight) {
              const size_t rowOffset = channelOffset + ih * rowStride;

              for (int kw = 0; kw < dims.kernelWidth; ++kw) {
                const int iw = ow * stride + kw - pad;

                // Check column validity and copy
                if (iw >= 0 && iw < dims.inputWidth) {
                  columns(patchIdx, colIdx) = inData[rowOffset + iw];
                } else {
                  columns(patchIdx, colIdx) = val_t(0);
                }
                ++patchIdx;
              }
            } else {
              // Entire row is out of bounds - fill with zeros
              for (int kw = 0; kw < dims.kernelWidth; ++kw) {
                columns(patchIdx++, colIdx) = val_t(0);
              }
            }
          }
        }
        ++colIdx;
      }
    }
  }
};

// N-dim Tensor (row-major, initializer_list indexer)
using tensor_t = TensorND;

// MISC UTILITIY FUNCTIONS
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
