#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

class MNISTData {
 public:
  explicit MNISTData(const string &filePath) { _load(filePath); }
  ~MNISTData() = default;

 public:
  vector<vector<val_t>> getImages() { return images; }

  // one-hot
  vector<vector<val_t>> getLabels() { return labels; }

  void print(int idx) {
    cout << "label: ";
    for (auto l : labels[idx]) {
      cout << l << " ";
    }
    cout << endl;

    const vector<val_t> &image = images[idx];
    string chars = ".#";
    cout << "image: " << endl;
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        int char_idx = (image[i * 28 + j] > 0 ? 1 : 0) * (chars.size() - 1);
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
      label[static_cast<int>(target)] = val_t(1);
      labels.push_back(label);

      vector<val_t> image(28 * 28);
      for (int i = 0; i < 28 * 28; i++) {
        image[i] = stof(values[i + 1]) / val_t(255);
      }
      images.push_back(image);
    }
  }

  std::vector<std::string> _split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }

 private:
  vector<vector<val_t>> images;
  vector<vector<val_t>> labels;
};
