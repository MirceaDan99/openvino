# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "moe_set_tensor_test")

# Source files
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Create executable
ov_add_sample(NAME ${TARGET_NAME}
              SOURCES ${SRC}
              DEPENDENCIES ie_samples_utils)

add_dependencies(${TARGET_NAME} openvino_intel_npu_plugin)