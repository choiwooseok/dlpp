#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class ImageDataset {
 public:
  ImageDataset() = default;
  virtual ~ImageDataset() = default;

 public:
  virtual void printData(int idx) = 0;
  virtual void load(const std::string& filePath) = 0;

  std::vector<std::vector<float>> getImages() {
    return images_;
  }

  // one-hot encoded labels
  std::vector<std::vector<float>> getLabels() {
    return labels_;
  }

  void clear() {
    images_.clear();
    labels_.clear();
  }

  std::vector<float> onehotEncode(int label, int numClasses) {
    std::vector<float> v(numClasses, 0.0f);
    v[label] = 1.0f;
    return v;
  }

  // csv read helper
  std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }

  void printLabel(const std::string& label, int val) {
    std::cout << std::format("label: {} ({})", label, val) << std::endl;
  }

  void printImage(const std::vector<float>& image, int h, int w, int numChannels) {
    const std::string chars = " .:-=+*#%@";

    std::cout << "image:" << std::endl;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        int charIdx = 0;

        if (numChannels == 1) {
          charIdx = static_cast<int>(image[i * h + j] * (chars.size() - 1));

        } else if (numChannels == 3) {
          float r = image[i * w + j];
          float g = image[h * w * 1 + i * w + j];
          float b = image[h * w * 2 + i * w + j];
          float gray = 0.299 * r + 0.587 * g + 0.114 * b;

          charIdx = static_cast<int>(gray * (chars.size() - 1));
        }

        std::cout << chars[charIdx] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

 protected:
  std::vector<std::vector<float>> images_;
  std::vector<std::vector<float>> labels_;
};