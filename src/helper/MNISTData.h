#pragma once

#include "ImageDataset.h"

class MNISTData : public ImageDataset {
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
  void printData(int idx) override {
    int label = std::distance(labels_[idx].begin(), std::ranges::max_element(labels_[idx]));
    printLabel(classToString(static_cast<Class>(label)), label);
    printImage(images_[idx], IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS);
  }

  void load(const std::string& filepath) override {
    std::fstream file(filepath);
    std::string line;
    while (getline(file, line)) {
      std::vector<std::string> values = split(line, ',');

      int label = stoi(values[0]);
      labels_.push_back(onehotEncode(label, NUM_CLASSES));

      std::vector<float> image(IMAGE_SIZE);
      for (int i = 0; i < IMAGE_SIZE; i++) {
        image[i] = stof(values[i + 1]) / 255.f;
      }
      images_.push_back(image);
    }
    file.close();
  }
};

class MNISTDataBuilder {
 public:
  MNISTDataBuilder& withTrainData() {
    data_.clear();
    data_.load("../resource/mnist/mnist_train.csv");
    return *this;
  }

  MNISTDataBuilder& withTestData() {
    data_.clear();
    data_.load("../resource/mnist/mnist_test.csv");
    return *this;
  }

  MNISTData build() {
    return data_;
  }

 private:
  MNISTData data_;
};