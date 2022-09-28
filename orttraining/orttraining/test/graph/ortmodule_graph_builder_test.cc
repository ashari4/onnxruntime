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
constexpr auto MATMUL_MODEL_PATH = ORT_TSTR("testdata/matmul_1.onnx");
}  // namespace


TEST(ORTModuleGraphBuilderTest, BuildMatMulGraphTest) {
    OrtModuleGraphBuilder graph_builder;
    OrtModuleGraphBuilderConfiguration config;
    std::ifstream model_stream(MATMUL_MODEL_PATH, std::ios::in | std::ios::binary);
    graph_builder.Initialize(model_stream, config);

    // assert that there is 1 Q and 1 DQ node that feeds into the MatMul.
}

}  // namespace test
}  // namespace onnxruntime
