// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class DequantizeBFP final : public OpKernel {
 public:
  DequantizeBFP(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

class QuantizeBFP final : public OpKernel {
 public:
  QuantizeBFP(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};
}  // namespace onnxruntime
