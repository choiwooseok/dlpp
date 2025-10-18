#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

#include "EigenHelper.h"

class MNISTData {
private:
  vector<vector<val_t>> images;
  vector<vector<val_t>> labels;

public:
  MNISTData(const string &filePath) { _load(filePath); }

public:
  Eigen::MatrixXf getImages() {
    return toEigenMatrix<val_t, vector<vector<val_t>>>(images);
  }

  Eigen::MatrixXf getLabels() {
    return toEigenMatrix<val_t, vector<vector<val_t>>>(labels);
  }

  void print(const Eigen::RowVectorXf &image, const Eigen::RowVectorXf &label) {
    cout << "label: ";
    for (auto l : label) {
      cout << l << " ";
    }
    cout << endl;

    string chars = ".#";

    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        int char_idx = image[i * 28 + j] * (chars.size() - 1);
        cout << chars[char_idx] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

private:
  void _load(const string &filepath) {
    images.clear();
    labels.clear();

    fstream file(filepath);
    string line;
    while (getline(file, line)) {
      vector<string> values = _split(line, ',');
      float target = stof(values[0]);
      vector<val_t> label(10);
      label[(int)target] = val_t(1);
      labels.push_back(label);

      vector<val_t> image(28 * 28);
      for (int i = 0; i < 28 * 28; i++) {
        image[i] = stof(values[i + 1]) > val_t(0) ? val_t(1) : val_t(0);
      }
      images.push_back(image);
    }
  };

  std::vector<std::string> _split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }
};
