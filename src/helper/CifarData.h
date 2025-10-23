#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include "EigenHelper.h"

enum class CIFAR10_CLASSES {
  AIRPLANE = 0,
  AUTOMOBILE = 1,
  BIRD = 2,
  CAT = 3,
  DEER = 4,
  DOG = 5,
  FROG = 6,
  HORSE = 7,
  SHIP = 8,
  TRUCK = 9
};

// Structure to hold a single CIFAR-10 image and its label
struct CifarImage {
  uint8_t label;
  std::vector<uint8_t> pixels; // 32x32x3 = 3072 bytes
};

class CifarData {
public:
  CifarData(const std::string &filePath) {
    images = readCifar10Binary(filePath);
    images = convertToGrayscale(images);
  }

  vector<uint8_t> getLabels() {
    vector<uint8_t> labels;
    for (auto &img : images) {
      labels.push_back(img.label);
    }
    return labels;
  }

  mat_t getImages() {
    vector<vector<uint8_t>> imgs;
    for (auto &img : images) {
      imgs.push_back(img.pixels);
    }
    return toEigenMatrix<uint8_t, vector<vector<uint8_t>>>(imgs);
  }

  void readCifar10Binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return images;
    }

    while (file.peek() != EOF) { // Check if there's more data to read
      CifarImage image;
      image.pixels.resize(3072); // Allocate space for 32x32x3 pixels

      // Read the 1-byte label
      file.read(reinterpret_cast<char *>(&image.label), sizeof(image.label));

      // Read the 3072-byte image data
      file.read(reinterpret_cast<char *>(image.pixels.data()),
                image.pixels.size());

      if (file.gcount() == (sizeof(image.label) + image.pixels.size())) {
        images.push_back(image);
      } else {
        // Handle incomplete reads, e.g., if the file is truncated
        std::cerr << "Warning: Incomplete record read from " << filename
                  << std::endl;
        break;
      }
    }

    file.close();
  }

  CifarImage convertToGrayscale(const CifarImage &rgb_image) {
    CifarImage graysacle_image;
    graysacle_image.label = rgb_image.label;
    grayscale_image.pixels.resize(32 * 32);

    const vector<uint8_t> &rgb_data = rgb_image.pixels;

    for (int i = 0; i < 32 * 32; ++i) {
      uint8_t r = rgb_data[i];
      uint8_t g = rgb_data[i + 32];
      uint8_t b = rgb_data[i + 2 * 32];

      // Apply luminosity formula: 0.21R + 0.72G + 0.07B
      grayscale_image.pixels[i] =
          static_cast<uint8_t>(0.21 * r + 0.72 * g + 0.07 * b);
    }
  }

private:
  std::vector<CifarImage> images;
};
