#pragma once

#include "ImageDataset.h"

class Cifar10Data : public ImageDataset {
 public:
  static constexpr int IMAGE_WIDTH = 32;
  static constexpr int IMAGE_HEIGHT = 32;
  static constexpr int NUM_CHANNELS = 3;
  static constexpr int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS;
  static constexpr int NUM_CLASSES = 10;
  static constexpr int IMAGES_PER_BATCH = 10000;

  enum class Class : int {
    airplane = 0,
    automobile,
    bird,
    cat,
    deer,
    dog,
    frog,
    horse,
    ship,
    truck,
  };

  static std::string classToString(Class c) {
    std::unordered_map<Class, std::string> lookup = {
        {Class::airplane, "airplane"},
        {Class::automobile, "automobile"},
        {Class::bird, "bird"},
        {Class::cat, "cat"},
        {Class::deer, "deer"},
        {Class::dog, "dog"},
        {Class::frog, "frog"},
        {Class::horse, "horse"},
        {Class::ship, "ship"},
        {Class::truck, "truck"},
    };

    return lookup[c];
  }

 public:
  Cifar10Data() = default;
  ~Cifar10Data() = default;

 public:
  void printData(int idx) override {
    int label = std::distance(labels_[idx].begin(), std::ranges::max_element(labels_[idx]));
    printLabel(classToString(static_cast<Class>(label)), label);
    printImage(images_[idx], IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS);
  }

  void load(const std::string& filePath) override {
    std::ifstream file(filePath, std::ios::binary);

    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filePath << std::endl;
      throw std::runtime_error("Failed to open CIFAR-10 batch file: " + filePath);
    }

    std::vector<char> buffer(1 + IMAGE_SIZE);

    for (int i = 0; i < IMAGES_PER_BATCH; ++i) {
      file.read(buffer.data(), buffer.size());

      if (!file) {
        std::cerr << "Warning: Could only read " << i << " images from " << filePath << std::endl;
        break;
      }

      // Read the 1-byte label -> one-hot encoding
      int label = static_cast<unsigned char>(buffer[0]);
      labels_.push_back(onehotEncode(label, NUM_CLASSES));

      // Read the 3072-byte image data (R, G, B channels in order)
      // CIFAR-10 format: [R1...R1024, G1...G1024, B1...B1024]
      std::vector<float> image(IMAGE_SIZE);
      for (int j = 0; j < IMAGE_SIZE; ++j) {
        image[j] = static_cast<float>(static_cast<unsigned char>(buffer[1 + j])) / 255.f;
      }
      images_.push_back(image);
    }
    file.close();
  }
};

class Cifar10DataBuilder {
 public:
  Cifar10DataBuilder& withTrainData() {
    data_.clear();
    for (int i = 0; i < 5; i++) {
      data_.load("../resource/cifar10/data_batch_" + std::to_string(i + 1) + ".bin");
    }
    return *this;
  }

  Cifar10DataBuilder& withTestData() {
    data_.clear();
    data_.load("../resource/cifar10/test_batch.bin");
    return *this;
  }

  Cifar10Data build() {
    return data_;
  }

 private:
  Cifar10Data data_;
};
