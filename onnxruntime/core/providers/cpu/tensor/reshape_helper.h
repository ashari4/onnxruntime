// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor_shape.h"

namespace onnxruntime {

// Verify and convert unknown dim during reshape
class ReshapeHelper {
 public:
  ReshapeHelper(const TensorShape& input_shape, TensorShapeVector& requested_shape, bool allow_zero = false) {
    const auto input_shape_size = input_shape.Size();
    ORT_ENFORCE(input_shape_size != -1,
                "The input tensor must not have any dynamic (-1) dimensions. Input shape:", input_shape);

    auto nDims = requested_shape.size();
    ptrdiff_t unknown_dim = -1;
    int64_t size = 1;
    for (size_t i = 0; i < nDims; ++i) {
      ORT_ENFORCE(requested_shape[i] >= -1, "A dimension cannot be less than -1, got ", requested_shape[i]);
      if (requested_shape[i] == -1) {
        ORT_ENFORCE(unknown_dim == -1, "At most one dimension can be -1.");
        unknown_dim = i;
      } else {
        if (!allow_zero && requested_shape[i] == 0) {
          ORT_ENFORCE(i < input_shape.NumDimensions(),
                      "The dimension with value zero exceeds"
                      " the dimension size of the input tensor.");
          requested_shape[i] = input_shape[i];
        }
        size *= requested_shape[i];
      }
    }

    if (unknown_dim != -1) {
      // calculate unknown dimension
      ORT_ENFORCE(size != 0 && (input_shape_size % size) == 0,
                  "The input tensor cannot be reshaped to the requested shape. Input shape:", input_shape,
                  ", requested shape:", TensorShape(requested_shape));
      requested_shape[unknown_dim] = input_shape_size / size;
    } else {
      // check if the output shape is valid.
      ORT_ENFORCE(input_shape_size == size,
                  "The input tensor cannot be reshaped to the requested shape. Input shape:", input_shape,
                  ", requested shape:", TensorShape(requested_shape));
    }
  }
};

}  // namespace onnxruntime
