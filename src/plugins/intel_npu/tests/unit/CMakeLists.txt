# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_npu_unit_tests")

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/partitioning
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/backend/src
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/utils/src
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/npuw
            ${CMAKE_CURRENT_SOURCE_DIR}/npuw/model_generator
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/utils/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/al/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/backend/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/common/include
            ${OpenVINO_SOURCE_DIR}/src/inference/src
            $<TARGET_PROPERTY:openvino_npu_zero_utils,INCLUDE_DIRECTORIES>
        OBJECT_FILES
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/al/src/config/npuw.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/al/src/config/config.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/common/src/npu.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/common/src/sync_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/async_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/remote_context.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/metadata.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model_utils.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/logging.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_prefix_caching.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/infer_request_utils.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/serialization.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/util.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/util_xarch.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/lazy_tensor.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/attention.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/pyramid_attention.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/host_flash_attention.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe_transformations/moe_transformation.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe_transformations/moe_unroll_patterns.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe_transformations/device_routed_moe_transform.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe_transformations/gather_to_2d_gather.cpp
        LINK_LIBRARIES
            common_test_utils_s
            openvino::reference
        DEPENDENCIES
            openvino_intel_npu_plugin
        LABELS
            OV UNIT NPU NPUW
)

if(ENABLE_AVX2)
    ov_avx2_optimization_flags(avx2_flags)
    target_compile_options(${TARGET_NAME} PRIVATE "${avx2_flags}")
    target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_AVX2)
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)
