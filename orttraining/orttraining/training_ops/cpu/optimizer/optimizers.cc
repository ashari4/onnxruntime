// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/optimizer/optimizers.h"

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status SGDOptimizer<T>::Compute(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& W = *ctx->Input<Tensor>(1);
  const Tensor& G = *ctx->Input<Tensor>(2);
  Tensor* NW = ctx->Output(0, W.Shape());
  Tensor* NG = ctx->Output(1, G.Shape());

  // NW = W - eta * G
  float eta = *ETA.template Data<float>();
  const auto& delta = -eta * MakeEigenArrayMap<T>(G);

  if (NG != nullptr) {
    MakeEigenArrayMap<T>(*NG) = delta;
  }
  if (NW != nullptr) {
    MakeEigenArrayMap<T>(*NW) = MakeEigenArrayMap<T>(W) + delta;
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    SGDOptimizer,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .Alias(1, 0)  // Update weights in-place
        .Alias(2, 1)  // Update gradients in-place
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SGDOptimizer<float>);

template <>
Status AdamOptimizer<BFloat16>::Compute(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& S = *ctx->Input<Tensor>(1);
  const Tensor& W = *ctx->Input<Tensor>(2);
  const Tensor& G = *ctx->Input<Tensor>(3);
  const Tensor& M1 = *ctx->Input<Tensor>(4);
  const Tensor& M2 = *ctx->Input<Tensor>(5);

  Tensor& NS = *ctx->Output(0, S.Shape());
  Tensor& NM1 = *ctx->Output(1, M1.Shape());
  Tensor& NM2 = *ctx->Output(2, M2.Shape());
  Tensor* NW = ctx->Output(3, W.Shape());
  Tensor* NG = ctx->Output(4, G.Shape());

  const float eta = *ETA.template Data<float>();
  const int64_t step = *S.template Data<int64_t>();

  const auto M1T = MakeEigenArrayMap<float>(M1);
  const auto M2T = MakeEigenArrayMap<float>(M2);
  const auto WT = MakeEigenArrayMap<float>(W);
  const auto GT = MakeEigenArrayMap<BFloat16>(G).cast<float>();

  // Update exponentially-averaged historical gradient
  MakeEigenArrayMap<float>(NM1) = alpha_ * M1T + ((1 - alpha_) * GT);

  // Update exponentially-averaged historical squared gradient
  MakeEigenArrayMap<float>(NM2) = beta_ * M2T + ((1 - beta_) * GT * GT);

  const float alpha_correction = do_bias_correction_ ?
    compute_bias_correction_coefficient(alpha_, step) : 1.f;
  const float beta_correction = do_bias_correction_ ?
    compute_bias_correction_coefficient(beta_, step) : 1.f;

  const auto NM1T = MakeEigenArrayMap<float>(NM1);
  const auto NM2T = MakeEigenArrayMap<float>(NM2);

  // Currently two modes of Adamw are supported:
  // Mode 0: Pytorch https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW,
  //         bias correction is applied on m and v individually,
  //         weight decay is applied before weight is updated.
  // Mode 1: Huggingface https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW.,
  //         bias correction is applied on learning rate,
  //         weight decay is applied after weight is updated.
  if(weight_decay_mode_ == 0) {
    // Compute weight update.
    const auto& denom = (NM2T / beta_correction).sqrt() + epsilon_;
    const auto& update = ( (NM1T / alpha_correction) / denom) + (lambda_ * WT);
    const auto& delta = -eta * update;

    // Weight, gradient, and step update.
    if (NG != nullptr) {
      MakeEigenArrayMap<BFloat16>(*NG) = delta.cast<BFloat16>();
    }
    if (NW != nullptr) {
      const auto temp = WT + delta;
      MakeEigenArrayMap<float>(*NW) = temp;
    }
  }
  else if (weight_decay_mode_ == 1) {
    const auto& denom = NM2T.sqrt() + epsilon_;
    const auto& step_size = eta * std::sqrt(beta_correction) / alpha_correction;

    // Huggingface updates weights in the following logic:
    // param' = param - step_size * m1o / denom
    // param_out = param' - original_lr * lambda * param'
    // then param_out = param - step_size * m1o / denom - original_lr * lambda * (param - step_size * m1o / denom)
    // so delta = -step_size * m1o / denom - original_lr * lambda * (param - step_size * m1o / denom)
    const auto& delta = -step_size * NM1T / denom
                        - eta * lambda_ * (WT - step_size * NM1T / denom);

    // Weight, gradient, and step update.
    if (NG != nullptr) {
      MakeEigenArrayMap<BFloat16>(*NG) = delta.cast<BFloat16>();
    }
    if (NW != nullptr) {
      const auto temp = WT + delta;
      MakeEigenArrayMap<float>(*NW) = temp;
    }
  }
  else {
    // Shouldn't reach here
    ORT_THROW("Unsupported Adamw optimizer mode.");
  }

  *NS.template MutableData<int64_t>() = step + 1;
  return Status::OK();
}


template <typename T>
Status AdamOptimizer<T>::Compute(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& S = *ctx->Input<Tensor>(1);
  const Tensor& W = *ctx->Input<Tensor>(2);
  const Tensor& G = *ctx->Input<Tensor>(3);
  const Tensor& M1 = *ctx->Input<Tensor>(4);
  const Tensor& M2 = *ctx->Input<Tensor>(5);

  Tensor& NS = *ctx->Output(0, S.Shape());
  Tensor& NM1 = *ctx->Output(1, M1.Shape());
  Tensor& NM2 = *ctx->Output(2, M2.Shape());
  Tensor* NW = ctx->Output(3, W.Shape());
  Tensor* NG = ctx->Output(4, G.Shape());

  const float eta = *ETA.template Data<float>();
  const int64_t step = *S.template Data<int64_t>();

  // Update exponentially-averaged historical gradient
  MakeEigenArrayMap<T>(NM1) = alpha_ * MakeEigenArrayMap<T>(M1) + ((1 - alpha_) * MakeEigenArrayMap<T>(G));

  // Update exponentially-averaged historical squared gradient
  MakeEigenArrayMap<T>(NM2) = beta_ * MakeEigenArrayMap<T>(M2) + ((1 - beta_) * MakeEigenArrayMap<T>(G) * MakeEigenArrayMap<T>(G));

  const float alpha_correction = do_bias_correction_ ?
    compute_bias_correction_coefficient(alpha_, step) : 1.f;
  const float beta_correction = do_bias_correction_ ?
    compute_bias_correction_coefficient(beta_, step) : 1.f;

  // Currently two modes of Adamw are supported:
  // Mode 0: Pytorch https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW,
  //         bias correction is applied on m and v individually,
  //         weight decay is applied before weight is updated.
  // Mode 1: Huggingface https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW.,
  //         bias correction is applied on learning rate,
  //         weight decay is applied after weight is updated.
  if(weight_decay_mode_ == 0) {
    // Compute weight update.
    const auto& denom = (MakeEigenArrayMap<T>(NM2) / beta_correction).sqrt() + epsilon_;
    const auto& update = ( (MakeEigenArrayMap<T>(NM1) / alpha_correction) / denom) + (lambda_ * MakeEigenArrayMap<T>(W));
    const auto& delta = -eta * update;

    // Weight, gradient, and step update.
    if (NG != nullptr) {
      MakeEigenArrayMap<T>(*NG) = delta;
    }
    if (NW != nullptr) {
      MakeEigenArrayMap<T>(*NW) = MakeEigenArrayMap<T>(W) + delta;
    }
  }
  else if (weight_decay_mode_ == 1) {
    const auto& denom = MakeEigenArrayMap<T>(NM2).sqrt() + epsilon_;
    const auto& step_size = eta * std::sqrt(beta_correction) / alpha_correction;

    // Huggingface updates weights in the following logic:
    // param' = param - step_size * m1o / denom
    // param_out = param' - original_lr * lambda * param'
    // then param_out = param - step_size * m1o / denom - original_lr * lambda * (param - step_size * m1o / denom)
    // so delta = -step_size * m1o / denom - original_lr * lambda * (param - step_size * m1o / denom)
    const auto& delta = -step_size * MakeEigenArrayMap<T>(NM1) / denom
                        - eta * lambda_ * (MakeEigenArrayMap<T>(W) - step_size * MakeEigenArrayMap<T>(NM1) / denom);

    // Weight, gradient, and step update.
    if (NG != nullptr) {
      MakeEigenArrayMap<T>(*NG) = delta;
    }
    if (NW != nullptr) {
      MakeEigenArrayMap<T>(*NW) = MakeEigenArrayMap<T>(W) + delta;
    }
  }
  else {
    // Shouldn't reach here
    ORT_THROW("Unsupported Adamw optimizer mode.");
  }

  *NS.template MutableData<int64_t>() = step + 1;
  return Status::OK();
}


#define REGISTER_ADAM_KERNEL_TYPED(T1, T2, T3, T4, T_GRAD)          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                       \
      AdamOptimizer,                                                                                   \
      kMSDomain,                                                                                       \
      1,                                                                                               \
      T1##_##T2##_##T3##_##T4##_##T_GRAD,                     \
      kCpuExecutionProvider,                                                                          \
      (*KernelDefBuilder::Create())                                                                    \
          .Alias(1, 0)                              /* Update step count in-place */                   \
          .Alias(2, 3)                              /* Update weights in-place */                      \
          .Alias(3, 4)                              /* Update gradients in-place */                    \
          .Alias(4, 1)                              /* Update moment-1 in-place */                     \
          .Alias(5, 2)                              /* Update moment-2 in-place */                     \
          .Alias(6, 5)                              /* Update mixed_precision weights in-place */      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                                     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())                                     \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())                                     \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>())                                     \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>()),                             \
      AdamOptimizer<T3>);

REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, float)
REGISTER_ADAM_KERNEL_TYPED(float, int64_t, float, float, BFloat16)
}  // namespace contrib
}  // namespace onnxruntime
