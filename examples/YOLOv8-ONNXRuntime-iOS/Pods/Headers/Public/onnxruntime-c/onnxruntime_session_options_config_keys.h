// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/*
 * This file defines SessionOptions Config Keys and format of the Config Values.
 *
 * The Naming Convention for a SessionOptions Config Key,
 * "[Area][.[SubArea1].[SubArea2]...].[Keyname]"
 * Such as "ep.cuda.use_arena"
 * The Config Key cannot be empty
 * The maximum length of the Config Key is 128
 *
 * The string format of a SessionOptions Config Value is defined individually for each Config.
 * The maximum length of the Config Value is 1024
 */

// Key for disable PrePacking,
// If the config value is set to "1" then the prepacking is disabled, otherwise prepacking is enabled (default value)
static const char* const kOrtSessionOptionsConfigDisablePrepacking = "session.disable_prepacking";

// A value of "1" means allocators registered in the env will be used. "0" means the allocators created in the session
// will be used. Use this to override the usage of env allocators on a per session level.
static const char* const kOrtSessionOptionsConfigUseEnvAllocators = "session.use_env_allocators";

// Set to 'ORT' (case sensitive) to load an ORT format model.
// If unset, model type will default to ONNX unless inferred from filename ('.ort' == ORT format) or bytes to be ORT
static const char* const kOrtSessionOptionsConfigLoadModelFormat = "session.load_model_format";

// Set to 'ORT' (case sensitive) to save optimized model in ORT format when SessionOptions.optimized_model_path is set.
// If unset, format will default to ONNX unless optimized_model_filepath ends in '.ort'.
static const char* const kOrtSessionOptionsConfigSaveModelFormat = "session.save_model_format";

// If a value is "1", flush-to-zero and denormal-as-zero are applied. The default is "0".
// When multiple sessions are created, a main thread doesn't override changes from succeeding session options,
// but threads in session thread pools follow option changes.
// When ORT runs with OpenMP, the same rule is applied, i.e. the first session option to flush-to-zero and
// denormal-as-zero is only applied to global OpenMP thread pool, which doesn't support per-session thread pool.
// Note that an alternative way not using this option at runtime is to train and export a model without denormals
// and that's recommended because turning this option on may hurt model accuracy.
static const char* const kOrtSessionOptionsConfigSetDenormalAsZero = "session.set_denormal_as_zero";

// It controls to run quantization model in QDQ (QuantizelinearDeQuantizelinear) format or not.
// "0": enable. ORT does fusion logic for QDQ format.
// "1": disable. ORT doesn't do fusion logic for QDQ format.
// Its default value is "0" unless the DirectML execution provider is registered, in which case it defaults to "1".
static const char* const kOrtSessionOptionsDisableQuantQDQ = "session.disable_quant_qdq";

// It controls whether to enable Double QDQ remover and Identical Children Consolidation
// "0": not to disable. ORT does remove the middle 2 Nodes from a Q->(QD->Q)->QD pairs
// "1": disable. ORT doesn't remove the middle 2 Nodes from a Q->(QD->Q)->QD pairs
// Its default value is "0"
static const char* const kOrtSessionOptionsDisableDoubleQDQRemover = "session.disable_double_qdq_remover";

// If set to "1", enables the removal of QuantizeLinear/DequantizeLinear node pairs once all QDQ handling has been
// completed. e.g. If after all QDQ handling has completed and we have -> FloatOp -> Q -> DQ -> FloatOp -> the
// Q -> DQ could potentially be removed. This will provide a performance benefit by avoiding going from float to
// 8-bit and back to float, but could impact accuracy. The impact on accuracy will be model specific and depend on
// other factors like whether the model was created using Quantization Aware Training or Post Training Quantization.
// As such, it's best to test to determine if enabling this works well for your scenario.
// The default value is "0"
// Available since version 1.11.
static const char* const kOrtSessionOptionsEnableQuantQDQCleanup = "session.enable_quant_qdq_cleanup";

// Enable or disable gelu approximation in graph optimization. "0": disable; "1": enable. The default is "0".
// GeluApproximation has side effects which may change the inference results. It is disabled by default due to this.
static const char* const kOrtSessionOptionsEnableGeluApproximation = "optimization.enable_gelu_approximation";

// This setting controls whether to enable AheadOfTime function inlining.
// AOT function inlining examines the graph and attempts to inline as many locally defined functions in the model
// as possible with the help of enabled execution providers.
// This can reduce the number of function calls and improve performance because it is done before
// Level1 optimizers and constant folding. However, under some circumstances, when the EPs are not available,
// one can disable the AOT inlining, produce an optimized model and postpone AOT until run time.
// "0": enable; "1": disable.
// Its default value is "0".
static const char* const kOrtSessionOptionsDisableAheadOfTimeFunctionInlining = "session.disable_aot_function_inlining";

#ifdef ENABLE_TRAINING
// Specifies a list of op types for memory footprint reduction.
// The value should be a ","-delimited list of pair of
// <subgraph string: optimization strategy: number of subgraph to apply>.
// For example, "Gelu+Cast+:1:0,Dropout+:1:1".
//   A valid "subgraph string" should be one subgraph representation output by ORT graph transformations.
//   "optimization strategy" currently has valid values: 0 - disabled, 1 - recompute.
//   "number of subgraph to apply" is used to control how many subgraphs to apply optimization, to avoid "oversaving"
//   the memory.
static const char* const kOrtSessionOptionsMemoryOptimizerEnabler = "optimization.memory_optimizer_config";

// Specifies the config for detecting subgraphs for memory footprint reduction.
// The value should be a string contains int separated using commas. The default value is "0:0".
static const char* const kOrtSessionOptionsMemoryOptimizerProbeConfig = "optimization.enable_memory_probe_recompute_config";
#endif

// Enable or disable using device allocator for allocating initialized tensor memory. "1": enable; "0": disable. The default is "0".
// Using device allocators means the memory allocation is made using malloc/new.
static const char* const kOrtSessionOptionsUseDeviceAllocatorForInitializers = "session.use_device_allocator_for_initializers";

// Configure whether to allow the inter_op/intra_op threads spinning a number of times before blocking
// "0": thread will block if found no job to run
// "1": default, thread will spin a number of times before blocking
static const char* const kOrtSessionOptionsConfigAllowInterOpSpinning = "session.inter_op.allow_spinning";
static const char* const kOrtSessionOptionsConfigAllowIntraOpSpinning = "session.intra_op.allow_spinning";

// Key for using model bytes directly for ORT format
// If a session is created using an input byte array contains the ORT format model data,
// By default we will copy the model bytes at the time of session creation to ensure the model bytes
// buffer is valid.
// Setting this option to "1" will disable copy the model bytes, and use the model bytes directly. The caller
// has to guarantee that the model bytes are valid until the ORT session using the model bytes is destroyed.
static const char* const kOrtSessionOptionsConfigUseORTModelBytesDirectly = "session.use_ort_model_bytes_directly";

/// <summary>
/// Key for using the ORT format model flatbuffer bytes directly for initializers.
/// This avoids copying the bytes and reduces peak memory usage during model loading and initialization.
/// Requires `session.use_ort_model_bytes_directly` to be true.
/// If set, the flatbuffer bytes provided when creating the InferenceSession MUST remain valid for the entire
/// duration of the InferenceSession.
/// </summary>
static const char* const kOrtSessionOptionsConfigUseORTModelBytesForInitializers =
    "session.use_ort_model_bytes_for_initializers";

// This should only be specified when exporting an ORT format model for use on a different platform.
// If the ORT format model will be used on ARM platforms set to "1". For other platforms set to "0"
// Available since version 1.11.
static const char* const kOrtSessionOptionsQDQIsInt8Allowed = "session.qdqisint8allowed";

// x64 SSE4.1/AVX2/AVX512(with no VNNI) has overflow problem with quantizied matrix multiplication with U8S8.
// To avoid this we need to use slower U8U8 matrix multiplication instead. This option, if
// turned on, use slower U8U8 matrix multiplications. Only effective with AVX2 or AVX512
// platforms.
static const char* const kOrtSessionOptionsAvx2PrecisionMode = "session.x64quantprecision";

// Specifies how minimal build graph optimizations are handled in a full build.
// These optimizations are at the extended level or higher.
// Possible values and their effects are:
// "save": Save runtime optimizations when saving an ORT format model.
// "apply": Only apply optimizations available in a minimal build.
// ""/<unspecified>: Apply optimizations available in a full build.
// Available since version 1.11.
static const char* const kOrtSessionOptionsConfigMinimalBuildOptimizations =
    "optimization.minimal_build_optimizations";

// Note: The options specific to an EP should be specified prior to appending that EP to the session options object in
// order for them to take effect.

// Specifies a list of stop op types. Nodes of a type in the stop op types and nodes downstream from them will not be
// run by the NNAPI EP.
// The value should be a ","-delimited list of op types. For example, "Add,Sub".
// If not specified, the default set of stop ops is used. To specify an empty stop ops types list and disable stop op
// exclusion, set the value to "".
static const char* const kOrtSessionOptionsConfigNnapiEpPartitioningStopOps = "ep.nnapi.partitioning_stop_ops";

// Enabling dynamic block-sizing for multithreading.
// With a positive value, thread pool will split a task of N iterations to blocks of size starting from:
// N / (num_of_threads * dynamic_block_base)
// As execution progresses, the size will decrease according to the diminishing residual of N,
// meaning the task will be distributed in smaller granularity for better parallelism.
// For some models, it helps to reduce the variance of E2E inference latency and boost performance.
// The feature will not function by default, specify any positive integer, e.g. "4", to enable it.
// Available since version 1.11.
static const char* const kOrtSessionOptionsConfigDynamicBlockBase = "session.dynamic_block_base";

// This option allows to decrease CPU usage between infrequent
// requests and forces any TP threads spinning stop immediately when the last of
// concurrent Run() call returns.
// Spinning is restarted on the next Run() call.
// Applies only to internal thread-pools
static const char* const kOrtSessionOptionsConfigForceSpinningStop = "session.force_spinning_stop";

// "1": all inconsistencies encountered during shape and type inference
// will result in failures.
// "0": in some cases warnings will be logged but processing will continue. The default.
// May be useful to expose bugs in models.
static const char* const kOrtSessionOptionsConfigStrictShapeTypeInference = "session.strict_shape_type_inference";

// "1": every model using a more recent opset than the latest released one will fail
// "0": the model may or may not work if onnxruntime cannot find an implementation, this option
// is used for development purpose.
static const char* const kOrtSessionOptionsConfigStrictAllowReleasedOpsetsOnly = "session.allow_released_opsets_only";

// The file saves configuration for partitioning node among logic streams
static const char* const kNodePartitionConfigFile = "session.node_partition_config_file";

// This Option allows setting affinities for intra op threads.
// Affinity string follows format:
// logical_processor_id,logical_processor_id;logical_processor_id,logical_processor_id
// Semicolon isolates configurations among threads, while comma split processors where ith thread expected to attach to.
// e.g.1,2,3;4,5
// specifies affinities for two threads, with the 1st thread attach to the 1st, 2nd, and 3rd processor, and 2nd thread to the 4th and 5th.
// To ease the configuration, an "interval" is also allowed:
// e.g. 1-8;8-16;17-24
// orders that the 1st thread runs on first eight processors, 2nd thread runs on next eight processors, and so forth.
// Note:
// 1. Once set, the number of thread affinities must equal to intra_op_num_threads - 1, since ort does not set affinity on the main thread which
//    is started and managed by the calling app;
// 2. For windows, ort will infer the group id from a logical processor id, for example, assuming there are two groups with each has 64 logical processors,
//    an id of 64 will be inferred as the last processor of the 1st group, while 65 will be interpreted as the 1st processor of the second group.
//    Hence 64-65 is an invalid configuration, because a windows thread cannot be attached to processors across group boundary.
static const char* const kOrtSessionOptionsConfigIntraOpThreadAffinities = "session.intra_op_thread_affinities";

// This option will dump out the model to assist debugging any issues with layout transformation,
// and is primarily intended for developer usage. It is only relevant if an execution provider that requests
// NHWC layout is enabled such as NNAPI, XNNPACK or QNN.
//
// Default is off. Set to "1" to enable.
//
// If modified by layout transformation the model will be dumped after these steps:
//   1) insertion of the layout transformation Transpose nodes
//   2) after those are optimized using the transpose optimizer,
//   3) after the L1 transformers are applied to the updated graph.
// The model will be saved to filename post_layout_transform_step_<step_number>.onnx.
static const char* const kDebugLayoutTransformation = "session.debug_layout_transformation";

// Graph nodes that are not supported by the execution providers (EPs) explicitly added to the session are
// assigned (i.e., "fallback") to the CPU EP by default.
//
// This option allows the user to disable the fallback of unsupported graph nodes to the CPU EP.
// If this option is set to "1", session creation will fail if the execution providers other than the CPU EP cannot
// fully support all of the nodes in the graph.
//
// It is invalid to set this option and explicitly add the CPU EP to the session. In this case, session creation
// will also fail with an error.
//
// Option values:
// - "0": CPU EP fallback is not disabled. [DEFAULT]
// - "1": CPU EP fallback is disabled.
static const char* const kOrtSessionOptionsDisableCPUEPFallback = "session.disable_cpu_ep_fallback";

// Use this config when serializing a large model after optimization to specify an external initializers file
static const char* const kOrtSessionOptionsOptimizedModelExternalInitializersFileName =
    "session.optimized_model_external_initializers_file_name";

// Use this config to control the minimum size of the initializer when externalizing it during serialization
static const char* const kOrtSessionOptionsOptimizedModelExternalInitializersMinSizeInBytes =
    "session.optimized_model_external_initializers_min_size_in_bytes";

// Enable EP context feature to dump the partitioned graph which includes the EP context into Onnx file.
// The dumped Onnx model with EP context can be used for future inference to avoid the EP graph partitioning/compile overhead.
// "0": disable. (default)
// "1": enable.
static const char* const kOrtSessionOptionEpContextEnable = "ep.context_enable";

// Specify the file path for the Onnx model which has EP context.
// Default to original_file_name_ctx.onnx if not specified
static const char* const kOrtSessionOptionEpContextFilePath = "ep.context_file_path";

// Flag to specify whether to dump the EP context into the Onnx model.
// "0": dump the EP context into separate file, keep the file name in the Onnx model.
// "1": dump the EP context into the Onnx model. (default).
static const char* const kOrtSessionOptionEpContextEmbedMode = "ep.context_embed_mode";

// Gemm fastmath mode provides fp32 gemm acceleration with bfloat16 based matmul.
// Option values:
// - "0": Gemm FastMath mode is not enabled. [DEFAULT]
// - "1": Gemm FastMath mode is enabled.
static const char* const kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16 = "mlas.enable_gemm_fastmath_arm64_bfloat16";
