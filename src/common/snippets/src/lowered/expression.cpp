// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/type.hpp"

namespace ov {
namespace snippets {
namespace lowered {

Expression::Expression(const std::shared_ptr<Node>& n, const std::shared_ptr<IShapeInferSnippetsFactory>& factory)
        : m_source_node{n}, m_emitter{nullptr},
        m_input_port_connectors{}, m_output_port_connectors{}, m_shapeInference(make_shape_inference(n, factory)) {
    m_input_port_descriptors.reserve(n->get_input_size());
    m_output_port_descriptors.reserve(n->get_output_size());
    for (const auto& input : n->inputs()) {
        m_input_port_descriptors.push_back(PortDescriptorUtils::get_port_descriptor_ptr(input));
    }
    for (const auto& output : n->outputs()) {
        m_output_port_descriptors.push_back(PortDescriptorUtils::get_port_descriptor_ptr(output));
    }
}

const PortConnectorPtr& Expression::get_input_port_connector(size_t i) const {
    OPENVINO_ASSERT(i < m_input_port_connectors.size(), "Failed to get input port connector: target input port must be less than input count!");
    return m_input_port_connectors[i];
}
const PortConnectorPtr& Expression::get_output_port_connector(size_t i) const {
    OPENVINO_ASSERT(i < m_output_port_connectors.size(), "Failed to get output port connector: target output port must be less than output count!");
    return m_output_port_connectors[i];
}

const PortDescriptorPtr& Expression::get_input_port_descriptor(size_t i) const {
    OPENVINO_ASSERT(i < m_input_port_descriptors.size(), "Failed to get input port descriptor: target input port must be less than input count!");
    return m_input_port_descriptors[i];
}
const PortDescriptorPtr& Expression::get_output_port_descriptor(size_t i) const {
    OPENVINO_ASSERT(i < m_output_port_descriptors.size(), "Failed to get output port descriptor: target output port must be less than output count!");
    return m_output_port_descriptors[i];
}

std::shared_ptr<Node> Expression::get_node() const {
    if (!m_source_node)
        OPENVINO_THROW("An attempt to get uninitialized node from lowered expression");
    return  m_source_node;
}

std::shared_ptr<Emitter> Expression::get_emitter() const {
    return m_emitter;
}

RegInfo Expression::get_reg_info() const {
    RegInfo reg_info;
    reg_info.first.reserve(m_input_port_descriptors.size());
    reg_info.second.reserve(m_output_port_descriptors.size());
    for (const auto& port : m_input_port_descriptors)
        reg_info.first.push_back(port->get_reg());
    for (const auto& port : m_output_port_descriptors)
        reg_info.second.push_back(port->get_reg());
    return reg_info;
}

void Expression::set_reg_info(RegInfo rinfo) {
    const auto& in = rinfo.first;
    const auto& out = rinfo.second;
    OPENVINO_ASSERT(m_input_port_descriptors.size() == in.size(), "Incorrect count of input physical registers");
    OPENVINO_ASSERT(m_output_port_descriptors.size() == out.size(), "Incorrect count of output physical registers");
    for (size_t i = 0; i < m_input_port_descriptors.size(); ++i) {
        m_input_port_descriptors[i]->set_reg(in[i]);
    }
    for (size_t i = 0; i < m_output_port_descriptors.size(); ++i) {
        m_output_port_descriptors[i]->set_reg(out[i]);
    }
}

void Expression::validate() const {
    OPENVINO_ASSERT(m_input_port_descriptors.size() == m_input_port_connectors.size(),
                    "The count of input ports and input port connectors must be equal");
    OPENVINO_ASSERT(m_output_port_descriptors.size() == m_output_port_connectors.size(),
                    "The count of output ports and output port connectors must be equal");
    OPENVINO_ASSERT(m_source_node != nullptr,
                    "The expression has null source node");
}

void Expression::replace_input(size_t port, PortConnectorPtr to) {
    OPENVINO_ASSERT(port < m_input_port_connectors.size(), "Failed to replace: target input port must be less than input count!");
    m_input_port_connectors[port] = std::move(to);
}

std::vector<size_t> Expression::get_loop_ids() const {
    return m_loop_ids;
}

void Expression::set_loop_ids(const std::vector<size_t>& loops) {
    std::unordered_set<size_t> s(loops.begin(), loops.end());
    OPENVINO_ASSERT(s.size() == loops.size(), "Loop IDs must be unique");
    m_loop_ids = loops;
}

ExpressionPort Expression::get_input_port(size_t i) {
    return ExpressionPort(this->shared_from_this(), ExpressionPort::Type::Input, i);
}

ExpressionPort Expression::get_output_port(size_t i) {
    return ExpressionPort(this->shared_from_this(), ExpressionPort::Type::Output, i);
}

void Expression::updateShapes() {
    OPENVINO_ASSERT(m_shapeInference, "Attempt to UpdateShapes without initialized shapeInference");
    IShapeInferSnippets::Result result;
    try {
        std::vector<VectorDimsRef> input_shapes;

        const auto& in_connectors = get_input_port_connectors();
        const auto& in_descriptors = get_input_port_descriptors();

        input_shapes.reserve(in_connectors.size());
        for (size_t i = 0; i < in_connectors.size(); i++) {
            const auto& src_port_desc = in_connectors[i]->get_source().get_descriptor_ptr();
            in_descriptors[i]->set_shape(src_port_desc->get_shape());
            // Note that input_shape is a reference, so we should always bind it to an object with a longer lifetime
            input_shapes.emplace_back(in_descriptors[i]->get_shape());
        }

        result = m_shapeInference->infer(input_shapes);
    }
    catch (const std::exception& exp) {
        OPENVINO_THROW("Shape inference of " + (get_node()->get_friendly_name()) + " failed: " + exp.what());
    }
    OPENVINO_ASSERT(result.status == ShapeInferStatus::success,
                    "Shape inference of " + (get_node()->get_friendly_name()) + " didn't return success status");
    const auto& out_descriptors = get_output_port_descriptors();
    OPENVINO_ASSERT(result.dims.size() == out_descriptors.size(), "shapeInference call returned invalid number of output shapes");
    for (size_t i = 0; i < out_descriptors.size(); i++)
        out_descriptors[i]->set_shape(result.dims[i]);
}

IOExpression::IOExpression(const std::shared_ptr<ov::opset1::Parameter>& par, int64_t index, const std::shared_ptr<IShapeInferSnippetsFactory>& factory)
        : Expression(par, factory), m_index(index), m_type{io_type::INPUT} {}
IOExpression::IOExpression(const std::shared_ptr<ov::opset1::Result>& res, int64_t index, const std::shared_ptr<IShapeInferSnippetsFactory>& factory)
        : Expression(res, factory), m_index(index), m_type{io_type::OUTPUT} {}

}// namespace lowered
}// namespace snippets
}// namespace ov
