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
    - Softmax
    - Tanh
- Optimizer
  - GD
  - SGD
  - Adam
  - CyclicLR
  - OneCycleLR

# Build
- prerequisite
  - conan2
  - cmake

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

```
./resource
├── cifar10
│   ├── batches.meta.txt
│   ├── data_batch_1.bin
│   ├── data_batch_2.bin
│   ├── data_batch_3.bin
│   ├── data_batch_4.bin
│   ├── data_batch_5.bin
│   ├── readme.html
│   └── test_batch.bin
├── mnist
    ├── mnist_test.csv
    └── mnist_train.csv
```

# Binaries
- for training
  - xor_train
  - minist_train_fc
  - minist_train_cnn
  - cifar10_train_cnn
- dlpp_test (gtest)
```
# see list
./dlpp_test --gtest_list_tests

XORTestSuite.
  XORTest
MNIST/MNISTFixture.
  MNISTTest/0  # GetParam() = "fc"
  MNISTTest/1  # GetParam() = "cnn"
CIFAR10/CIFAR10Fixture.
  CIFAR10/0  # GetParam() = "cnn"
```

- run with filters
```
./dlpp_test --gtest_filter="*XOR*"
./dlpp_test --gtest_filter="*CIFAR10/0*"
./dlpp_test --gtest_filter="*MNISTTest/0*"
./dlpp_test --gtest_filter="*MNISTTest/1*"
```

# 3rd party
- @see here [conanfile.txt](conanfile.txt)
- nlohmann_json for serialization
- gtest for test
- Eigen for matrix calc