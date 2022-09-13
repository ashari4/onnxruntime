// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/quantization/quantize_bfp.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/qmath.h"

// todo: ah remove this
#include <iostream>

namespace onnxruntime {

Status DequantizeBFP::Compute(OpKernelContext* /* ctx */) const {
  std::cout << "in DequantizeBFP::Compute" << std::endl;
  // todo: ah create fp32 output tensor
  return Status::OK();
}

Status QuantizeBFP::Compute(OpKernelContext* /* ctx */) const {
  std::cout << "in QuantizeBFP::Compute" << std::endl;
  // todo: ah create uint8 tensor, shape and stride
  return Status::OK();
}
}  // namespace onnxruntime
