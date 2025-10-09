#pragma once

#include <filesystem>
#include <string>

class ResourceManager {
 public:
  static ResourceManager& instance() {
    static ResourceManager instance;
    return instance;
  }

  void setModelDir(const std::filesystem::path& dir) {
    modelDir_ = dir;
  }

  std::filesystem::path getModelPath(const std::string& filename) const {
    return modelDir_ / filename;
  }

 private:
  std::filesystem::path modelDir_ = "../resource/model/";
};