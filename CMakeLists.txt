cmake_minimum_required(VERSION 3.15)
project(dlpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)

###################
#    PROJ LIB     #
###################
add_library(${PROJECT_NAME}
    src/MLP.cpp
)
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -funroll-loops)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    nlohmann_json::nlohmann_json
)

###################
#   MNIST TRAIN   #
###################
add_executable(mnist_train
    src/executables/mnist_train.cpp
)
target_link_libraries(mnist_train 
    PRIVATE 
    ${PROJECT_NAME}
)

###################
#    XOR TRAIN    #
###################
add_executable(xor_train
    src/executables/xor_train.cpp
)
target_link_libraries(xor_train 
    PRIVATE 
    ${PROJECT_NAME}
)

##################
#    G - TEST    #
##################
add_executable(${PROJECT_NAME}_test
    test/test_main.cpp
    test/xor_test.cpp
    test/mnist_test.cpp
)

target_link_libraries(${PROJECT_NAME}_test 
    PRIVATE 
    ${PROJECT_NAME}
    gtest::gtest
)
