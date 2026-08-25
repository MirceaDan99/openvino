// Copyright (C) 2018-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string_view>

#include "openvino/core/runtime_attribute.hpp"

namespace intel_npu {

/**
 * @brief Attribute containing the memory address of a weights buffer and the size of the buffer in bytes.
 * @details Used as part of the model marshalling process to avoid the need of copying weights.
 */
class WeightsMmapAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("WeightsMmapAttribute", "0", RuntimeAttribute);

    WeightsMmapAttribute() = delete;

    WeightsMmapAttribute(const char* weights_path_pointer, const size_t offset)
        : weights_path_pointer(reinterpret_cast<size_t>(weights_path_pointer)),
          offset(offset) {}

    /**
     * @note The names of the attributes have been kept short in order to save some memory (there may be a lot of
     * "ov::Constant" nodes in a model). While deserializing, the name of the attribute ("WeightsPointerAttribute") is
     * also used as part of identification in order to avoid collision.
     */
    static constexpr const std::string_view WEIGHTS_PATH_POINTER_KEY = "mwpp";
    static constexpr const std::string_view OFFSET_KEY = "mo";

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute(WEIGHTS_PATH_POINTER_KEY.data(), weights_path_pointer);
        visitor.on_attribute(OFFSET_KEY.data(), offset);
        return true;
    }

    bool is_deterministic() const override {
        return false;
    }

    size_t weights_path_pointer;
    size_t offset;
};

}  // namespace intel_npu
