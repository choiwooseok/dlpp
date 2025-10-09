#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

class MNISTData {
 public:
  static constexpr int IMAGE_WIDTH = 28;
  static constexpr int IMAGE_HEIGHT = 28;
  static constexpr int NUM_CHANNELS = 1;
  static constexpr int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
  static constexpr int NUM_CLASSES = 10;

  enum class Class : int {
    zero = 0,
    one,
    two,
    three,
    four,
    five,
    six,
    seven,
    eight,
    nine,
  };

  static std::string classToString(Class c) {
    std::unordered_map<Class, std::string> lookup = {
        {Class::zero, "0"},
        {Class::one, "1"},
        {Class::two, "2"},
        {Class::three, "3"},
        {Class::four, "4"},
        {Class::five, "5"},
        {Class::six, "6"},
        {Class::seven, "7"},
        {Class::eight, "8"},
        {Class::nine, "9"},
    };

    return lookup[c];
  }

 public:
  MNISTData() = default;
  ~MNISTData() = default;

 public:
  std::vector<std::vector<float>> getImages() { return images_; }

  // one-hot encoded labels
  std::vector<std::vector<float>> getLabels() { return labels_; }

  void printData(int idx) {
    if (idx < 0 || idx >= static_cast<int>(images_.size())) {
      std::cerr << "Invalid image index: " << idx << std::endl;
      ;
      return;
    }

    std::cout << "label: ";
    const auto &label = labels_[idx];
    for (int i = 0; i < NUM_CLASSES; ++i) {
      if (label[i] > 0.5f) {
        std::cout << i << std::endl;
        break;
      }
    }

    const std::vector<float> &image = images_[idx];

    // ASCII characters for different intensities
    const std::string chars = " .:-=+*#%@";

    std::cout << "image: " << std::endl;
    for (int h = 0; h < IMAGE_HEIGHT; ++h) {
      for (int w = 0; w < IMAGE_WIDTH; ++w) {
        int charIdx = static_cast<int>(image[h * IMAGE_HEIGHT + w] * (chars.size() - 1));
        std::cout << chars[charIdx] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  void load(const std::string &filepath) {
    images_.clear();
    labels_.clear();

    std::fstream file(filepath);
    std::string line;
    while (getline(file, line)) {
      std::vector<std::string> values = _split(line, ',');

      int labelIdx = stoi(values[0]);
      labels_.push_back(_onehotEncode(labelIdx));

      std::vector<float> image(IMAGE_SIZE);
      for (int i = 0; i < IMAGE_SIZE; i++) {
        image[i] = stof(values[i + 1]) / 255.f;
      }
      images_.push_back(image);
    }
  }

 private:
  std::vector<float> _onehotEncode(int idx) {
    std::vector<float> v(NUM_CLASSES, 0.0f);
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
  std::vector<std::vector<float>> images_;
  std::vector<std::vector<float>> labels_;
};

class MNISTDataBuilder {
 public:
  MNISTDataBuilder &withTrainData() {
    data_.load("../resource/mnist/mnist_train.csv");
    return *this;
  }

  MNISTDataBuilder &withTestData() {
    data_.load("../resource/mnist/mnist_test.csv");
    return *this;
  }

  MNISTData build() {
    return data_;
  }

 private:
  MNISTData data_;
};