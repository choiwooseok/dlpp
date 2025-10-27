# Neural Network implmentation from Scratch

- for personal study
- impl w/o framework

# Modules
- Network
- Layers
  - FC
  - Conv2D
  - Pooling
    - Avg
    - Max
  - Flatten
  - Activations
    - LRelu
    - Relu
    - Sigmoid
- Optimizer
  - GD
  - SGD
  - Adam

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
- minist_train_fc
- minist_train_conv
- xor_train
- dlpp_test (Google Test)
  - XORTEST
  - MNIST
    - M_FC
    - M_CNN

```
./dlpp_test --gtest_filter="*XORTest*" 
./dlpp_test --gtest_filter="*M_FC*" 
./dlpp_test --gtest_filter="*M_CNN*" 
```

# 3rd party
- @see here [conanfile.txt](conanfile.txt)
- nlohmann_json for save & load
- gtest for test
- Eigen for matrix calc