#pragma once

#include <exception>
#include <string>

class LayerException : public std::runtime_error {
 public:
  explicit LayerException(const std::string& layer, const std::string& msg)
      : std::runtime_error(layer + ": " + msg) {}
};
