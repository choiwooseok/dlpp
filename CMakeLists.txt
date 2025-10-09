cmake_minimum_required(VERSION 3.15)
project(dlpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
add_compile_options(-Ofast)

set(DEP_LIBS 
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)
###################
#  CIFAR10 TRAIN  #
###################
add_executable(cifar10_train_cnn train/cifar10_train_cnn.cpp)
target_link_libraries(cifar10_train_cnn PRIVATE ${DEP_LIBS})

###################
#   MNIST TRAIN   #
###################
set(MNIST_MODELS "fc;cnn")
foreach(MNIST_MODEL IN LISTS MNIST_MODELS)
    add_executable(mnist_train_${MNIST_MODEL} train/mnist_train_${MNIST_MODEL}.cpp)
    target_link_libraries(mnist_train_${MNIST_MODEL} PRIVATE ${DEP_LIBS})
endforeach()

###################
#    XOR TRAIN    #
###################
add_executable(xor_train train/xor_train.cpp)
target_link_libraries(xor_train PRIVATE ${DEP_LIBS})

##################
#    G - TEST    #
##################
add_executable(${PROJECT_NAME}_test
    test/test_main.cpp
    test/xor_test.cpp
    test/mnist_test.cpp
    test/cifar10_test.cpp
)

target_link_libraries(${PROJECT_NAME}_test 
    PRIVATE 
    ${DEP_LIBS}
    gtest::gtest
)

