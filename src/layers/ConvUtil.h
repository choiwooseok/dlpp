#pragma once

#include "types.h"

// Convolution dimension information
struct ConvDimensions {
  int batchSize;
  int channels;
  int inputHeight;
  int inputWidth;
  int kernelHeight;
  int kernelWidth;
  int stride;
  int pad;
  int outputHeight;
  int outputWidth;
  int patchSize;
  int numColumns;
};

// Image to Column utility for efficient convolution operations
class ConvUtil {
 private:
  ConvUtil() = delete;

 public:
  // Compute output dimensions for convolution operation
  static ConvDimensions getDimensions(const Tensor& input, int kernelHeight, int kernelWidth, int stride, int pad) {
    const int batchSize = static_cast<int>(input.shape(0));
    const int channels = static_cast<int>(input.shape(1));
    const int inputHeight = static_cast<int>(input.shape(2));
    const int inputWidth = static_cast<int>(input.shape(3));
    const int outputHeight = (inputHeight + 2 * pad - kernelHeight) / stride + 1;
    const int outputWidth = (inputWidth + 2 * pad - kernelWidth) / stride + 1;

    return ConvDimensions{.batchSize = batchSize,
        .channels = channels,
        .inputHeight = inputHeight,
        .inputWidth = inputWidth,
        .kernelHeight = kernelHeight,
        .kernelWidth = kernelWidth,
        .stride = stride,
        .pad = pad,
        .outputHeight = outputHeight,
        .outputWidth = outputWidth,
        .patchSize = channels * kernelHeight * kernelWidth,
        .numColumns = outputHeight * outputWidth};
  }

  // Transform image region to column for single sample
  // Input format: (N, C, H, W)
  // Output format: (patchSize, numColumns) where patchSize = C*kH*kW
  static mat_t im2col(const Tensor& input, size_t sampleIdx, const ConvDimensions& dims) {
    mat_t columns(dims.patchSize, dims.numColumns);

    // Direct pointer access for performance
    const val_t* inData = input.data();
    const size_t sampleOffset = sampleIdx * input.strides(0);

    // Precompute strides
    const size_t channelStride = input.strides(1);
    const size_t rowStride = input.strides(2);

    int colIdx = 0;

    // Iterate over output spatial positions
    for (int oh = 0; oh < dims.outputHeight; ++oh) {
      for (int ow = 0; ow < dims.outputWidth; ++ow) {
        int patchIdx = 0;

        // Iterate over each channel
        for (int c = 0; c < dims.channels; ++c) {
          const size_t channelOffset = sampleOffset + c * channelStride;

          // Iterate over kernel spatial positions
          for (int kh = 0; kh < dims.kernelHeight; ++kh) {
            const int ih = oh * dims.stride + kh - dims.pad;

            // Early check for row validity
            if (ih >= 0 && ih < dims.inputHeight) {
              const size_t rowOffset = channelOffset + ih * rowStride;

              for (int kw = 0; kw < dims.kernelWidth; ++kw) {
                const int iw = ow * dims.stride + kw - dims.pad;

                // Check column validity and copy
                if (iw >= 0 && iw < dims.inputWidth) {
                  columns(patchIdx, colIdx) = inData[rowOffset + iw];
                } else {
                  columns(patchIdx, colIdx) = val_t(0);
                }
                ++patchIdx;
              }
            } else {
              // Entire row is out of bounds - fill with zeros
              for (int kw = 0; kw < dims.kernelWidth; ++kw) {
                columns(patchIdx++, colIdx) = val_t(0);
              }
            }
          }
        }
        ++colIdx;
      }
    }

    return columns;
  }

  // Transform column back to image region for single sample (backward pass)
  // Accumulates gradients into the output tensor
  static void col2im(Tensor& output, const mat_t& columns, size_t sampleIdx, const ConvDimensions& dims) {
    // Direct pointer access for performance
    val_t* outData = output.data();
    const val_t* colData = columns.data();
    const size_t sampleOffset = sampleIdx * output.strides(0);

    // Precompute strides
    const size_t channelStride = output.strides(1);
    const size_t rowStride = output.strides(2);

    int colIdx = 0;

    // Iterate over output spatial positions
    for (int oh = 0; oh < dims.outputHeight; ++oh) {
      for (int ow = 0; ow < dims.outputWidth; ++ow) {
        int patchIdx = 0;

        // Iterate over each channel
        for (int c = 0; c < dims.channels; ++c) {
          const size_t channelOffset = sampleOffset + c * channelStride;

          // Iterate over kernel spatial positions
          for (int kh = 0; kh < dims.kernelHeight; ++kh) {
            const int ih = oh * dims.stride + kh - dims.pad;

            // Check row validity
            if (ih >= 0 && ih < dims.inputHeight) {
              const size_t rowOffset = channelOffset + ih * rowStride;

              for (int kw = 0; kw < dims.kernelWidth; ++kw) {
                const int iw = ow * dims.stride + kw - dims.pad;

                // Check column validity and accumulate gradient
                if (iw >= 0 && iw < dims.inputWidth) {
                  outData[rowOffset + iw] += colData[patchIdx * dims.numColumns + colIdx];
                }
                ++patchIdx;
              }
            } else {
              // Skip entire row if out of bounds
              patchIdx += dims.kernelWidth;
            }
          }
        }
        ++colIdx;
      }
    }
  }
};