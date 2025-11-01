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
  - BatchNorm
  - Dropout
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

# Data ( Train / Test )
- locate to resource/
  - MNIST CSV from https://github.com/phoebetronic/mnist
  - Cifar10 from https://www.cs.toronto.edu/~kriz/cifar.html

# Binaries
- for training
  - xor_train
  - minist_train_fc
  - minist_train_conv
  - cifar10_train
- dlpp_test (gtest)
```
# see list
./dlpp_test --gtest_list_tests

XORTestSuite.
  XORTest
CIFAR10Fixture.
  CIFAR10
MNIST/MNISTFixture.
  MNISTTest/0  # GetParam() = "fc"
  MNISTTest/1  # GetParam() = "cnn"
```

- run with filters
```
./dlpp_test --gtest_filter="*XOR*"
./dlpp_test --gtest_filter="*CIFAR10*"
./dlpp_test --gtest_filter="*MNISTTest/0*"
./dlpp_test --gtest_filter="*MNISTTest/1*"
```

# 3rd party
- @see here [conanfile.txt](conanfile.txt)
- nlohmann_json for serialization
- gtest for test
- Eigen for matrix calc