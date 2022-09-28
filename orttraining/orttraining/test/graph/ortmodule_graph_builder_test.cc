// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/framework/test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/common/path_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/ortmodule_graph_builder.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/test/session/training_session_test_utils.h"

#include <fstream>

using namespace onnxruntime::logging;
using namespace onnxruntime::training;
using namespace google::protobuf::util;
using namespace onnxruntime::path_utils;
using namespace onnxruntime::test::training_session_test_utils;

namespace onnxruntime {
namespace test {

namespace {
constexpr auto MATMUL_MODEL_PATH = ORT_TSTR("testdata/matmul_1_ir_13.onnx");
}  // namespace


TEST(ORTModuleGraphBuilderTest, BuildMatMulGraphTest) {
    OrtModuleGraphBuilder graph_builder;

    OrtModuleGraphBuilderConfiguration config;
    config.initializer_names = {"W"};
    config.initializer_names_to_train = {"W"};
    config.input_names_require_grad = {"X"};
    config.build_gradient_graph = true;
    config.loglevel = logging::Severity::kVERBOSE;

    std::ifstream model_stream(MATMUL_MODEL_PATH, std::ios::in | std::ios::binary);
    graph_builder.Initialize(model_stream, config);

    // todo: ah set the bw ops in the config
    // format: "<op name>,<BFP>,n,<input_name:bfp_type:block_dim>,..."
    std::string dx_qconfig = "Gemm,BFP,2,Y_grad:0:1,W:0:1";
    std::string dw_qconfig = "Gemm,BFP,2,X:0:0,Y_grad:0:0";
    config.backward_ops_to_quantize = {dx_qconfig, dw_qconfig};
    graph_builder.Build();

    // todo: ah remove this. this is for debugging purposes.
    auto model_serialized = graph_builder.GetModel();
    std::ofstream out("grad_model.onnx", std::ios::binary);
    out << model_serialized;
    out.close();

    // assert that there is 1 Q and 1 DQ node that feeds into the MatMul.
}

}  // namespace test
}  // namespace onnxruntime
