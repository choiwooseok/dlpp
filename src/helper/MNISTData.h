#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

class MNISTData {
 public:
  static constexpr int IMAGE_WIDTH = 28;
  static constexpr int IMAGE_HEIGHT = 28;
  static constexpr int NUM_CHANNELS = 1;
  static constexpr int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
  static constexpr int NUM_CLASSES = 10;

 public:
  explicit MNISTData(const string &filePath) { _load(filePath); }
  ~MNISTData() = default;

 public:
  vector<vector<float>> getImages() { return images_; }

  // one-hot encoded labels
  vector<vector<float>> getLabels() { return labels_; }

  void printImage(int idx) {
    if (idx < 0 || idx >= static_cast<int>(images_.size())) {
      std::cerr << "Invalid image index: " << idx << std::endl;
      return;
    }

    cout << "label: ";
    const auto &label = labels_[idx];
    for (int i = 0; i < NUM_CLASSES; ++i) {
      if (label[i] > 0.5f) {
        cout << i << endl;
        break;
      }
    }

    const vector<float> &image = images_[idx];

    // ASCII characters for different intensities
    const string chars = " .:-=+*#%@";

    cout << "image: " << endl;
    for (int h = 0; h < IMAGE_HEIGHT; ++h) {
      for (int w = 0; w < IMAGE_WIDTH; ++w) {
        int charIdx = static_cast<int>(image[h * IMAGE_HEIGHT + w] * (chars.size() - 1));
        cout << chars[charIdx] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

 private:
  void _load(const string &filepath) {
    images_.clear();
    labels_.clear();

    fstream file(filepath);
    string line;
    while (getline(file, line)) {
      vector<string> values = _split(line, ',');

      int labelIdx = stoi(values[0]);
      labels_.push_back(_onehotEncode(labelIdx));

      vector<float> image(IMAGE_SIZE);
      for (int i = 0; i < IMAGE_SIZE; i++) {
        image[i] = stof(values[i + 1]) / 255.f;
      }
      images_.push_back(image);
    }
  }

  vector<float> _onehotEncode(int idx) {
    vector<float> v(NUM_CLASSES, 0.0f);
    v[idx] = 1.0f;
    return v;
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
  vector<vector<float>> images_;
  vector<vector<float>> labels_;
};
