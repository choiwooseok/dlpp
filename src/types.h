#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <numeric>
#include <random>
#include <vector>

enum class INIT {
  NONE,
  XAVIER,
  HE,
};

using val_t = float;
using vec_t = Eigen::VectorXf;
using mat_t = Eigen::MatrixXf;
using RowMajorMatrix = Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// N-dimensional Tensor
class Tensor {
 public:
  // Factory methods
  static Tensor fromFlat(const vec_t& v, const std::vector<size_t>& shape) {
    Tensor T(shape);
    T.data_ = v;
    return T;
  }

  static Tensor fromMat(const mat_t& M) {
    if (M.size() == 0) {
      return Tensor();
    }

    const size_t rows = M.rows();
    const size_t cols = M.cols();

    Tensor T({rows, cols});

    for (auto r : std::views::iota(0UL, rows)) {
      for (auto c : std::views::iota(0UL, cols)) {
        T[r * cols + c] = M(r, c);
      }
    }

    return T;
  }

 public:
  Tensor() = default;
  ~Tensor() = default;
  Tensor(Tensor&& other) noexcept
      : data_(std::move(other.data_)), strides_(std::move(other.strides_)), shape_(std::move(other.shape_)) {}

  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      strides_ = std::move(other.strides_);
      shape_ = std::move(other.shape_);
    }
    return *this;
  }
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;

  explicit Tensor(const std::vector<size_t>& shape)
      : shape_(shape) {
    _calcStrides();
    _allocateData();
  }

 public:
  // Dimension queries
  size_t strides(int idx) const {
    return strides_[idx];
  }

  // number of dimensions
  size_t dim() const {
    return shape_.size();
  }

  const std::vector<size_t>& shape() const {
    return shape_;
  }
  size_t shape(int idx) const {
    return shape_[idx];
  }

  // total elements size
  size_t size() const {
    return shape_.empty() ? 0 : std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
  }

  bool empty() const {
    return size() == 0;
  }

  void printShape() const {
    std::cout << "shape: [ ";
    std::copy(shape_.cbegin(), shape_.cend(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "]" << std::endl;
  }

  int max_element_idx() const {
    auto it = std::ranges::max_element(data_);
    return std::distance(data_.begin(), it);
  }

  const vec_t& flatten() const {
    return data_;
  }

  // Element access - initializer_list
  inline val_t& operator()(std::initializer_list<size_t> idxs) {
    return data_[static_cast<int>(_calcIndex(idxs))];
  }

  inline val_t operator()(std::initializer_list<size_t> idxs) const {
    return data_[static_cast<int>(_calcIndex(idxs))];
  }

  // Element access - explicit 4D
  inline val_t& at(size_t n, size_t c, size_t h, size_t w) {
    return data_[static_cast<int>(n * strides(0) + c * strides(1) + h * strides(2) + w * strides_[3])];
  }

  inline val_t at(size_t n, size_t c, size_t h, size_t w) const {
    return data_[static_cast<int>(n * strides(0) + c * strides(1) + h * strides(2) + w * strides_[3])];
  }

  // Flat access
  inline val_t& operator[](size_t i) {
    return data_[static_cast<int>(i)];
  }

  inline val_t operator[](size_t i) const {
    return data_[static_cast<int>(i)];
  }

  // pointer access
  constexpr val_t* data() noexcept {
    return data_.data();
  }
  constexpr const val_t* data() const noexcept {
    return data_.data();
  }

  // Matrix mapping
  auto asMatrixConst(size_t rows, size_t cols) const {
    return Eigen::Map<const RowMajorMatrix>(data_.data(), static_cast<int>(rows), static_cast<int>(cols));
  }

  auto asMatrix(size_t rows, size_t cols) {
    return Eigen::Map<RowMajorMatrix>(data_.data(), static_cast<int>(rows), static_cast<int>(cols));
  }

  void fill(val_t value) {
    if (!empty()) {
      data_.setConstant(value);
    }
  }

  // In-place reshape (modifies current tensor)
  void reshape(const std::vector<size_t>& newShape) {
    shape_ = newShape;
    _calcStrides();
  }

 private:
  void _calcStrides() {
    const size_t n = dim();
    if (n == 0) {
      return;
    }

    strides_.resize(n);
    strides_[n - 1] = 1;
    for (int i = static_cast<int>(dim()) - 1; i > 0; --i) {
      strides_[i - 1] = strides_[i] * shape_[i];
    }
  }

  void _allocateData() {
    const size_t s = size();
    if (s > 0) {
      data_ = vec_t::Zero(s);
    }
  }

  size_t _calcIndex(const std::vector<size_t>& idxs) const {
    return std::inner_product(idxs.begin(), idxs.end(), shape_.begin(), 0UL);
  }

 private:
  vec_t data_;
  std::vector<size_t> strides_;
  std::vector<size_t> shape_;
};

// Eigen Helpers
inline static std::vector<val_t> toStdVector(const vec_t& vec) {
  return std::vector<val_t>(vec.data(), vec.data() + vec.size());
}

inline static vec_t toEigenVector(const std::vector<val_t>& vec) {
  return vec_t::Map(vec.data(), vec.size());
}

inline static mat_t toEigenMatrix(const std::vector<std::vector<val_t>>& vec) {
  mat_t M(vec.size(), vec.front().size());
  for (size_t i = 0; i < vec.size(); i++) {
    for (size_t j = 0; j < vec.front().size(); j++) {
      M(i, j) = vec[i][j];
    }
  }
  return M;
}

inline static std::vector<std::vector<val_t>> toStd2DVector(const mat_t& M) {
  std::vector<std::vector<val_t>> result(M.rows(), std::vector<val_t>(M.cols()));
  for (auto i : std::views::iota(0, static_cast<int>(M.rows()))) {
    for (auto j : std::views::iota(0, static_cast<int>(M.cols()))) {
      result[i][j] = M(i, j);
    }
  }
  return result;
}

// MISC UTILITIY FUNCTIONS
static val_t genRandom() {
  std::random_device rd;
  std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<> dis(val_t(-1), val_t(1));
  return dis(gen);
}

static long long timePointToMillis(const std::chrono::steady_clock::time_point& tp) {
  using namespace std::chrono;
  return duration_cast<milliseconds>(tp.time_since_epoch()).count();
}

static long long getCurrentTimeMillis() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
