// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/primitives/cum_sum.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCumSumOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::CumSum>& op) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto exclusive = op->is_exclusive();
    auto reverse = op->is_reverse();

    int64_t axis = 0;
    if (op->get_input_size() == 2) {
        auto axes_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(axes_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        axis = axes_constant->cast_vector<int64_t>()[0];
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    axis = ov::normalize_axis(op.get(), axis, op->get_input_partial_shape(0).rank());
    OPENVINO_SUPPRESS_DEPRECATED_END

    auto primitive = cldnn::cum_sum(layerName,
                                    inputs[0],
                                    axis,
                                    exclusive,
                                    reverse);

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(v0, CumSum);

}  // namespace intel_gpu
}  // namespace ov
