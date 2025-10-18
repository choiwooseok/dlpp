#pragma once

#include <Eigen/Eigen>
#include <vector>

template <typename Scalar, typename Vector>
inline static std::vector<Scalar> fromEigenVector(const Vector &vec) {
  return std::vector<Scalar>(vec.data(), vec.data() + vec.size());
}

template <typename Scalar, typename Container>
inline static Eigen::Matrix<Scalar, -1, 1>
toEigenVector(const vector<Scalar> &vec) {
  return Eigen::Matrix<Scalar, -1, 1>::Map(vec.data(), vec.size());
}

template <typename Scalar, typename Container>
inline static Eigen::Matrix<Scalar, -1, -1>
toEigenMatrix(const Container &vectors) {
  typedef typename Container::value_type VectorType;
  Eigen::Matrix<Scalar, -1, -1> M(vectors.size(), vectors.front().size());
  for (size_t i = 0; i < vectors.size(); i++)
    for (size_t j = 0; j < vectors.front().size(); j++)
      M(i, j) = vectors[i][j];
  return M;
}

template <typename Scalar, typename Matrix>
inline static std::vector<std::vector<Scalar>>
fromEigenMatrix(const Matrix &M) {
  std::vector<std::vector<Scalar>> m;
  m.resize(M.rows(), std::vector<Scalar>(M.cols(), 0));
  for (size_t i = 0; i < m.size(); i++)
    for (size_t j = 0; j < m.front().size(); j++)
      m[i][j] = M(i, j);
  return m;
}