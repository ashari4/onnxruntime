// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/quantization/quantize_bfp.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/qmath.h"

// todo: ah remove this
#include <algorithm>
#include <iostream>

namespace onnxruntime {

Status DequantizeBFP::Compute(OpKernelContext* ctx) const {
  std::cout << "in DequantizeBFP::Compute" << std::endl;
  auto& shape = *ctx->Input<Tensor>(1);
  // todo: ah use the stride
  [[maybe_unused]] auto& stride = *ctx->Input<Tensor>(2);
  TensorShape output_shape(shape.Data<int64_t>(), shape.Shape().Size());
  auto& output = ctx->RequiredOutput(0, output_shape);

  auto& raw_bfp = *ctx->Input<Tensor>(0);
  auto begin = raw_bfp.Data<uint8_t>();
  auto end = begin + raw_bfp.SizeInBytes();
  std::copy(begin, end, reinterpret_cast<uint8_t *>(output.MutableDataRaw()));
  return Status::OK();
}

Status QuantizeBFP::Compute(OpKernelContext* ctx) const {
  std::cout << "in QuantizeBFP::Compute" << std::endl;
  const auto& input = *ctx->Input<Tensor>(0);
  auto size = input.SizeInBytes();
  // for demo purposes, bfp tesor is just a copy of the original tensor
  auto& raw_bfp = ctx->RequiredOutput(0, {static_cast<int64_t>(size)});
  auto begin = reinterpret_cast<const uint8_t*>(input.DataRaw());
  std::copy(begin, begin + size, raw_bfp.MutableData<uint8_t>());

  auto& input_shape = input.Shape();
  auto& shape = ctx->RequiredOutput(1, {static_cast<int64_t>(input_shape.NumDimensions())});
  input.Shape().CopyDims(shape.MutableData<int64_t>(), input_shape.NumDimensions());
  [[maybe_unused]] auto& stride = ctx->RequiredOutput(2, {static_cast<int64_t>(input_shape.NumDimensions())});
#ifdef ENABLE_TRAINING
  // todo: ah copy stride data from input
#else
  // stride data not available - assume the output is contiguous.
  // todo: ah set the strides to be contiguous.
#endif
  return Status::OK();
}
}  // namespace onnxruntime
