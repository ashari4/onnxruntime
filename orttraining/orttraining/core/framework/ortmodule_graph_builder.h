// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_set>
#include <unordered_map>

#include "core/common/status.h"
#include "core/graph/model.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {
namespace training {

/**
 * The training configuration options.
 */
struct OrtModuleGraphBuilderConfiguration {
  // The names of the weights.
  std::vector<std::string> initializer_names{};
  // The names of the weights to train.
  std::vector<std::string> initializer_names_to_train{};
  // The names of inputs that require gradient.
  std::vector<std::string> input_names_require_grad{};

  // If the backward operator appears in this list, quantize its inputs.
  // string format: "<op name>,<BFP>,<input_name:block_dim>,..."
  // If BFP is specified, then QuantizeBFP and DequantizeBFP nodes are added.
  // In the future, other types of quantization like Linear can be added.
  // If the quantization type is BFP, then block_dim specifies the block dimension for the operand with name input_name
  std::vector<std::string> backward_ops_to_quantize{};

  // Graph configuration.
  bool use_memory_efficient_gradient = false;
  bool build_gradient_graph = true;
  bool enable_caching = false;

  // Graph transformer configuration
  TrainingGraphTransformerConfiguration graph_transformer_config{};

  // Log severity
  logging::Severity loglevel{logging::Severity::kWARNING};
};

/**
 * The information of graphs for frontend.
 */
struct GraphInfo {
  // The user inputs.
  std::vector<std::string> user_input_names{};
  // Map from user input names to corresponding user input grad names for those user inputs that require grad.
  std::unordered_map<std::string, std::string> user_input_grad_names{};
  // All initializers (trainable as well as non trainable).
  std::vector<std::string> initializer_names{};
  // Trainable initializers.
  std::vector<std::string> initializer_names_to_train{};
  // Trainable initializer grad names, ordered according to initializer_names_to_train.
  std::vector<std::string> initializer_grad_names_to_train{};
  // The user outputs.
  std::vector<std::string> user_output_names{};
  // Indices of output grads that are non-differentiable.
  std::vector<size_t> output_grad_indices_non_differentiable{};
  // Indices of output grads that need to be materialized to full size all-0 tensor.
  // Otherwise, we can use scalar-0 tensor.
  std::vector<size_t> output_grad_indices_require_full_shape{};
  // Indices of module output that are needed for backward computation
  std::vector<size_t> module_output_indices_requires_save_for_backward{};
  // Names of module outputs' gradient
  std::vector<std::string> module_output_gradient_name{};
  // Names of the frontier tensor corresponding to param
  std::unordered_map<std::string, std::string> frontier_node_arg_map{};
  // Names of the frontier NodeArgs in the order in which they will
  // be retrieved in the forward pass
  std::vector<std::string> cached_node_arg_names{};
};

class OrtModuleGraphBuilder {
 public:
  /**
   * Initialize the builder. It saves the initial model and the configuration.
   * It also removes the trainable initializers from initial model and move them to graph inputs.
   * @param model_istream The initial model as input stream.
   * @param config The configuration to control the builder.
   * @return The status of the initialization.
   */
  Status Initialize(std::istream& model_istream, const OrtModuleGraphBuilderConfiguration& config);

  /**
   * Optimize the inference graph and build the gradient graph.
   * @param input_shapes_ptr The pointer to vector of concrete shapes of the user inputs.
   * @return The status of the optimizing and building the gradient graph.
   */
  Status Build(const std::vector<std::vector<int64_t>>* input_shapes_ptr = nullptr);

  /**
   * Get inference/gradient model.
   * @return The optimized inference/gradient model serialized to string.
   */
  std::string GetModel() const;

  /**
   * Get inference optimized model.
   * @return The gradient model serialized to string.
   */
  std::string GetInferenceOptimizedModel() const;

  /**
   * Get the graphs information.
   * @return The graphs information.
   */
  GraphInfo GetGraphInfo() const { return graph_info_; }

 private:
 struct BFPConfig
 {
  int64_t bfp_type;
  int64_t block_dim;
 };
 using InputQConfigs = std::unordered_map<std::string, BFPConfig>;

  // Set concrete shapes for graph inputs.
  void SetConcreteInputShapes(const std::vector<std::vector<int64_t>>& input_shapes);

  // Apply graph transformers
  Status OptimizeInferenceGraph(std::unordered_set<std::string>& x_node_arg_names);

  // Build gradient graph.
  Status BuildGradientGraph(const std::unordered_set<std::string>& x_node_arg_names);

  // Get the "frontier" tensors- the the output of series of operations
  // that only depend on the param values, eg Casting a param
  void GetFrontierTensors();

  // Handle user outputs and output grads.
  void HandleOutputsAndGrads();

  // Reorder gradient graph outputs.
  void ReorderOutputs();

  // Add Q and DQ operators for nodes that accept quantized inputs.
  void AddQDQ();

  // Find the module output that are needed for backward computation
  void FindModuleOutputNeededForBackward();

  // Update require grad info for PythonOp.
  void UpdatePythonOpInputsRequireGradInfo(
      const std::unordered_map<std::string, std::vector<int64_t>>& python_op_input_require_grad_info);

  std::shared_ptr<onnxruntime::Model> model_;
  std::shared_ptr<onnxruntime::Model> inference_optimized_model_;
  std::shared_ptr<onnxruntime::Model> gradient_model_;
  GraphInfo graph_info_;
  std::unordered_map<std::string, InputQConfigs> q_configs_;

  OrtModuleGraphBuilderConfiguration config_;
  const logging::Logger* logger_ = &logging::LoggingManager::DefaultLogger();  // use default logger for now.
};

}  // namespace training
}  // namespace onnxruntime
