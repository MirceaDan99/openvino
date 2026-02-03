// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>
#include <string_view>

#include "dev/core_impl.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "zero_infer_request.hpp"

namespace {

// Create L0 tensor with random data
ov::SoPtr<ov::IRemoteTensor> create_random_l0_tensor(const ov::SoPtr<ov::IRemoteContext>& dev_remote_context,
                                         const ov::element::Type& type,
                                         const ov::Shape& shape) {
    // Calculate tensor size
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }
    size_t byte_size = num_elements * type.size();

    // Create remote tensor
    auto dev_remote_tensor = dev_remote_context->create_tensor(type, shape);

    // Fill with random data (using host tensor as intermediate)
    // For 4-bit types (nf4, u4, i4), we need to work with raw bytes
    if (type == ov::element::nf4 || type == ov::element::u4 || type == ov::element::i4) {
        // Calculate byte size for packed 4-bit data (2 values per byte)
        size_t packed_byte_size = (num_elements + 1) / 2;
        
        // Create a buffer for random 4-bit packed data
        std::vector<uint8_t> random_data(packed_byte_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 15);
        
        // Pack random 4-bit values (2 per byte)
        for (size_t i = 0; i < num_elements; ++i) {
            uint8_t value = static_cast<uint8_t>(dis(gen) & 0x0F);
            if (i % 2 == 0) {
                random_data[i / 2] = value;  // Lower 4 bits
            } else {
                random_data[i / 2] |= (value << 4);  // Upper 4 bits
            }
        }
        
        // Create host tensor and copy raw bytes to it
        ov::Tensor host_tensor(type, shape);
        std::memcpy(host_tensor.data(), random_data.data(), packed_byte_size);
        
        // Copy to remote tensor
        auto host_tensor_impl = ov::get_tensor_impl(host_tensor);
        dev_remote_tensor->copy_from(host_tensor_impl._ptr);
        return dev_remote_tensor;
    }

    ov::Tensor host_tensor(type, shape);
    
    if (type == ov::element::f32) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        float* data = host_tensor.data<float>();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = dis(gen);
        }
    } else if (type == ov::element::f16) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        ov::float16* data = host_tensor.data<ov::float16>();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = ov::float16(dis(gen));
        }
    } else if (type == ov::element::i32) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int32_t> dis(-100, 100);
        int32_t* data = host_tensor.data<int32_t>();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = dis(gen);
        }
    } else if (type == ov::element::i64) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dis(-100, 100);
        int64_t* data = host_tensor.data<int64_t>();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = dis(gen);
        }
    } else if (type == ov::element::u8 || type == ov::element::i8) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(-128, 127);
        int8_t* data = host_tensor.data<int8_t>();
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = static_cast<int8_t>(dis(gen));
        }
    }

    // Copy to remote tensor (this is the initial data setup, not counted in set_tensor performance)
    auto host_tensor_impl = ov::get_tensor_impl(host_tensor);
    dev_remote_tensor->copy_from(host_tensor_impl._ptr);

    return dev_remote_tensor;
}

}

namespace ov {

namespace test {

namespace behavior {

class ZeroInferRequestTests : public ov::test::behavior::OVPluginTestBase {
protected:
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> init_struct;
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::Config npu_config = ::intel_npu::Config(options);
    const std::string_view target_device = "NPU";
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();

        init_struct = ::intel_npu::ZeroInitStructsHolder::getInstance();
    }

    void TearDown() override {
        init_struct = nullptr;
        APIBaseTest::TearDown();
    }
};

}  // namespace behavior

}  // namespace test

}  // namespace ov

using namespace ov::test::behavior;

TEST_F(ZeroInferRequestTests, CheckOptimizationsForChainedUpdateGraphArguments) {
    const size_t tensor_pool_size = 30;
    const size_t num_iterations = 10000;
    const std::string_view model_path = "Model0_kv1152_02_REP0108_moe_chunk_0.blob";
    std::ifstream blob_file(model_path, std::ios::binary);

    ov::CoreImpl dev_core;
    auto dev_remote_context = dev_core.get_default_context(target_device.data());
    auto dev_compiled_model = dev_core.import_model(blob_file, dev_remote_context, {});
    auto npu_compiled_model = std::dynamic_pointer_cast<intel_npu::ICompiledModel>(dev_compiled_model._ptr);
    OPENVINO_ASSERT(npu_compiled_model != nullptr, "Compiled model couldn't be casted to `intel_npu::ICompiledModel`!");

    intel_npu::ZeroInferRequest zero_infer_request(init_struct, npu_compiled_model, npu_config);

    std::vector<std::vector<ov::SoPtr<ov::IRemoteTensor>>> input_tensor_pools;
    for (size_t i = 0; i < dev_compiled_model->inputs().size(); ++i) {
        const auto& input = dev_compiled_model->inputs()[i];
        
        std::vector<ov::SoPtr<ov::IRemoteTensor>> tensor_pool;
        for (size_t j = 0; j < tensor_pool_size; ++j) {
            auto remote_tensor = create_random_l0_tensor(
                dev_remote_context,
                input.get_element_type(),
                input.get_shape()
            );
            tensor_pool.push_back(remote_tensor);
        }
        input_tensor_pools.push_back(tensor_pool);
    }

    size_t tensor_idx = 1;

    for (size_t iter = 0; iter < num_iterations; ++iter) {
        std::map<ov::Output<const ov::Node>, ov::SoPtr<ov::ITensor>> ports_tensors;
        for (size_t i = 0; i < dev_compiled_model->inputs().size(); ++i) {
            ports_tensors.insert({dev_compiled_model->inputs()[i], input_tensor_pools[i][tensor_idx]});
            tensor_idx = ++tensor_idx % 30;
        }
        zero_infer_request.set_tensors(ports_tensors);
        zero_infer_request.infer();
    }
}
