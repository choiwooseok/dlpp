#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

class Cifar10Data {
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
  void setResourceDir(const std::string& resourceDir) { rscDir_ = resourceDir; }

  std::vector<std::vector<float>> getPixels() { return pixels_; }

  // one-hot encoded labels
  std::vector<std::vector<float>> getLabels() { return labels_; }

  size_t getNumSamples() const { return pixels_.size(); }

  // Read all 5 training batches
  void readAllTrainData() {
    pixels_.clear();
    labels_.clear();

    std::cout << "Loading CIFAR-10 training data..." << std::endl;

    for (int i = 0; i < 5; i++) {
      std::string filename = rscDir_ + "data_batch_" + std::to_string(i + 1) + ".bin";
      std::cout << "  Reading: " << filename << std::endl;
      _readBatch(filename);
    }

    std::cout << "Total training samples loaded: " << pixels_.size() << std::endl;
  }

  // Read test batch
  void readTestData() {
    pixels_.clear();
    labels_.clear();

    std::cout << "Loading CIFAR-10 test data..." << std::endl;
    std::string fileName = rscDir_ + "test_batch.bin";
    std::cout << "  Reading: " << fileName << std::endl;
    _readBatch(fileName);
    std::cout << "Total test samples loaded: " << pixels_.size() << std::endl;
  }

  void printData(int idx) {
    if (idx < 0 || idx >= static_cast<int>(pixels_.size())) {
      std::cerr << "Invalid image index: " << idx << std::endl;
      return;
    }

    std::cout << "label: ";
    const auto& label = labels_[idx];
    for (int i = 0; i < NUM_CLASSES; ++i) {
      if (label[i] > 0.5f) {
        std::cout << classToString(static_cast<Class>(i)) << " (" << i << ")" << std::endl;
        break;
      }
    }

    const std::vector<float>& image = pixels_[idx];

    // ASCII characters for different intensities
    const std::string chars = " .:-=+*#%@";

    std::cout << "Grayscale image:" << std::endl;
    for (int h = 0; h < IMAGE_HEIGHT; ++h) {
      for (int w = 0; w < IMAGE_WIDTH; ++w) {
        float r = image[h * IMAGE_WIDTH + w];
        float g = image[1024 + h * IMAGE_WIDTH + w];
        float b = image[2048 + h * IMAGE_WIDTH + w];
        float gray = 0.299 * r + 0.587 * g + 0.114 * b;

        int charIdx = static_cast<int>(gray * (chars.size() - 1));
        std::cout << chars[charIdx] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

 private:
  // Read a single CIFAR-10 binary batch file
  void _readBatch(const std::string& filePath) {
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
      int labelIdx = static_cast<unsigned char>(buffer[0]);

      if (labelIdx < 0 || labelIdx >= NUM_CLASSES) {
        std::cerr << "Warning: Invalid label " << labelIdx << " at image " << i << std::endl;
        continue;
      }

      labels_.push_back(_onehotEncode(labelIdx));

      // Read the 3072-byte image data (R, G, B channels in order)
      // CIFAR-10 format: [R1...R1024, G1...G1024, B1...B1024]
      std::vector<float> pixels(IMAGE_SIZE);
      for (int j = 0; j < IMAGE_SIZE; ++j) {
        pixels[j] = static_cast<float>(static_cast<unsigned char>(buffer[1 + j])) / 255.f;
      }
      pixels_.push_back(pixels);
    }

    file.close();
  }

  std::vector<float> _onehotEncode(int idx) {
    std::vector<float> v(NUM_CLASSES, 0.0f);
    v[idx] = 1.0f;
    return v;
  }

 private:
  std::vector<std::vector<float>> pixels_;
  std::vector<std::vector<float>> labels_;

  std::string rscDir_;
};

class Cifar10DataBuilder {
 public:
  Cifar10DataBuilder() {
    data_.setResourceDir("../resource/cifar10/");
  }

  Cifar10DataBuilder& withTrainData() {
    data_.readAllTrainData();
    return *this;
  }

  Cifar10DataBuilder& withTestData() {
    data_.readTestData();
    return *this;
  }

  Cifar10Data build() {
    return data_;
  }

 private:
  Cifar10Data data_;
};
