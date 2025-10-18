# Neural Network implmentation from Scratch

- for personal study
- impl w/o framework

# Modules
- Network
- Layers
  - FC
  - LRelu
  - Relu
  - Sigmoid
  - ... TODO add others

# Build
- [build.sh](build.sh)
```sh
#! /bin/sh

conan install . --output-folder=build --build=missing
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j12
```

# MNIST CSV from
* https://github.com/phoebetronic/mnist

# Executables
- minist_train
- xor_train
- dlpp_test (Google Test)
  - MNISTTEST
  - XORTEST

# 3rd party
- @see here [conanfile.txt](conanfile.txt)
- nlohmann_json for save & load
- gtest for test
- Eigen for matrix calc