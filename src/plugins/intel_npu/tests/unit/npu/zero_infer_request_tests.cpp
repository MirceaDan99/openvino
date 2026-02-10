// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <random>
#include <string_view>

#include "async_infer_request.hpp"
#include "dev/core_impl.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/sync_infer_request.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "remote_context.hpp"
#include "zero_backend.hpp"
#include "zero_infer_request.hpp"

namespace {

// Create L0 tensor with random data
ov::SoPtr<ov::IRemoteTensor> create_random_l0_tensor(const ov::SoPtr<intel_npu::RemoteContextImpl>& npu_remote_context,
                                                     const ov::element::Type& type,
                                                     const ov::Shape& shape) {
    // Calculate tensor size
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }

    // Create remote tensor
    auto npu_remote_context_ptr = std::make_shared<intel_npu::RemoteContextImpl>(
        *npu_remote_context);  // copy to align ZeroMemPool singleton to tests' executable, otherwise will differ from
                               // NPU's shared library
    auto npu_remote_tensor = npu_remote_context_ptr->create_tensor(type, shape, {});

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
        npu_remote_tensor->copy_from(host_tensor_impl._ptr);
        return npu_remote_tensor;
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
    npu_remote_tensor->copy_from(host_tensor_impl._ptr);

    return npu_remote_tensor;
}

}  // namespace

namespace ov {

namespace unit_test {

namespace intel_npu {

class ZeroInferRequestTests : public ::testing::Test {
protected:
    const std::string_view target_device = "NPU";
};

}  // namespace intel_npu

}  // namespace unit_test

}  // namespace ov

using namespace ov::unit_test::intel_npu;

TEST_F(ZeroInferRequestTests, CheckOptimizationsForChainedUpdateGraphArguments) {
    const size_t tensor_pool_size = 30;
    const size_t num_iterations = 1000;
    const std::string_view model_path = "Model0_kv1152_02_REP0108_moe_chunk_0.blob";
    std::ifstream blob_file(model_path.data(), std::ios::binary);

    auto dev_core_ptr = std::make_shared<ov::CoreImpl>();
    dev_core_ptr->register_compile_time_plugins();
    auto dev_remote_context = dev_core_ptr->get_default_context(target_device.data());
    auto npu_remote_context = std::dynamic_pointer_cast<intel_npu::RemoteContextImpl>(
        dev_remote_context._ptr);  // align ZeroMemPool singleton to tests' executable, otherwise will differ from NPU's
                                   // shared library
    OPENVINO_ASSERT(npu_remote_context != nullptr,
                    "Remote context couldn't be casted to `intel_npu::RemoteContextImpl`!");
    auto dev_compiled_model = dev_core_ptr->import_model(blob_file, dev_remote_context, {});
    auto dev_infer_request = dev_compiled_model->create_infer_request();
    auto npu_async_infer_request = std::dynamic_pointer_cast<intel_npu::AsyncInferRequest>(dev_infer_request);
    OPENVINO_ASSERT(npu_async_infer_request != nullptr,
                    "Infer request couldn't be casted to `intel_npu::AsyncInferRequest`!");
    auto npu_sync_infer_request = npu_async_infer_request->get_sync_infer_request();
    auto zero_infer_request = std::dynamic_pointer_cast<intel_npu::ZeroInferRequest>(npu_sync_infer_request);
    OPENVINO_ASSERT(zero_infer_request != nullptr,
                    "Sync infer request couldn't be casted to `intel_npu::ZeroInferRequest`!");

    std::vector<std::vector<ov::SoPtr<ov::IRemoteTensor>>> input_tensor_pools;
    for (size_t i = 0; i < dev_compiled_model->inputs().size(); ++i) {
        const auto& input = dev_compiled_model->inputs()[i];

        std::vector<ov::SoPtr<ov::IRemoteTensor>> tensor_pool;
        for (size_t j = 0; j < tensor_pool_size; ++j) {
            auto remote_tensor =
                create_random_l0_tensor(npu_remote_context, input.get_element_type(), input.get_shape());
            tensor_pool.push_back(remote_tensor);
        }
        input_tensor_pools.push_back(tensor_pool);
    }

    size_t tensor_idx = 1;

    // warmup
    auto zeroApiInstance =
        intel_npu::ZeroApi::getInstance();  // keep ZeroApi instance ownership inside current executable, the instance
                                            // from NPU's library will no longer be accessible and each `getInstance`
                                            // call gets the ZeroApi object allocated/deallocated loosing time
    zero_infer_request->infer();
    // warmup-end
    double diff_set_tensors_duration = .0;
    double diff_set_tensor_duration = .0;
    double diff_infer_duration = .0;

    std::cout << "Wait to setup VTune..." << std::endl;
    getchar();

    for (size_t iter = 0; iter < num_iterations; ++iter) {
        std::map<ov::Output<const ov::Node>, ov::SoPtr<ov::ITensor>> ports_tensors;
        for (size_t i = 0; i < dev_compiled_model->inputs().size(); ++i) {
            auto checkpoint_set_tensors_start{std::chrono::high_resolution_clock::now()};
            ports_tensors.insert({dev_compiled_model->inputs()[i], input_tensor_pools[i][tensor_idx]});
            auto checkpoint_set_tensors_stop{std::chrono::high_resolution_clock::now()};
            diff_set_tensors_duration += std::chrono::duration_cast<std::chrono::microseconds>(
                                             checkpoint_set_tensors_stop - checkpoint_set_tensors_start)
                                             .count();
            tensor_idx = ++tensor_idx % 30;
        }
        auto checkpoint_set_tensors_start{std::chrono::high_resolution_clock::now()};
        zero_infer_request->set_tensors(ports_tensors);
        auto checkpoint_set_tensors_stop{std::chrono::high_resolution_clock::now()};
        diff_set_tensors_duration += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint_set_tensors_stop -
                                                                                           checkpoint_set_tensors_start)
                                         .count();

        for (size_t i = 0; i < dev_compiled_model->inputs().size(); ++i) {
            auto checkpoint_set_tensor_start{std::chrono::high_resolution_clock::now()};
            zero_infer_request->set_tensor(dev_compiled_model->inputs()[i], input_tensor_pools[i][tensor_idx]);
            auto checkpoint_set_tensor_stop{std::chrono::high_resolution_clock::now()};
            diff_set_tensor_duration += std::chrono::duration_cast<std::chrono::microseconds>(
                                            checkpoint_set_tensor_stop - checkpoint_set_tensor_start)
                                            .count();
            tensor_idx = ++tensor_idx % 30;
        }

        auto checkpoint_infer_start{std::chrono::high_resolution_clock::now()};
        zero_infer_request->infer();
        auto checkpoint_infer_stop{std::chrono::high_resolution_clock::now()};
        diff_infer_duration +=
            std::chrono::duration_cast<std::chrono::microseconds>(checkpoint_infer_stop - checkpoint_infer_start)
                .count();
    }

    double avg_set_tensors_duration = diff_set_tensors_duration / 1000.0 / num_iterations;
    double avg_set_tensor_duration = diff_set_tensor_duration / 1000.0 / num_iterations;
    double avg_infer_duration = diff_infer_duration / 1000.0 / num_iterations;
    std::cout << "Summary for " << num_iterations << " iterations:" << std::endl;
    std::cout << std::fixed << "\tavg_set_tensors_duration = " << avg_set_tensors_duration << " ms" << std::endl;
    std::cout << std::fixed << "\tavg_set_tensor_duration = " << avg_set_tensor_duration << " ms" << std::endl;
    std::cout << std::fixed << "\tavg_infer_duration = " << avg_infer_duration << " ms" << std::endl;
    std::cout << std::fixed
              << "\t\tavg set_tensors / infer raport = " << avg_set_tensors_duration / avg_infer_duration * 100 << " %"
              << std::endl;
    std::cout << std::fixed
              << "\t\tavg set_tensor / infer raport = " << avg_set_tensor_duration / avg_infer_duration * 100 << " %"
              << std::endl;
}

TEST_F(ZeroInferRequestTests, CheckOptimizationsForNPUWMoePipeline) {
    ov::Core core;
    const std::string_view model_path = "";
    auto model = core.read_model(model_path.data());
    auto compiled_model = core.compile_model(model,
                                             target_device.data(),
                                             {ov::intel_npu::use_npuw(true),
                                              ov::intel_npu::npuw::llm::enabled(true),
                                              ov::intel_npu::npuw::partitioning::moe_token_chunk_size(0),
                                              ov::intel_npu::npuw::partitioning::moe_pool_size(8),
                                              ov::intel_npu::npuw::llm::prefill_moe_hint("HOST_ROUTED"),
                                              ov::intel_npu::npuw::llm::generate_moe_hint("HOST_ROUTED")});
    std::stringstream ss;
    compiled_model.export_model(ss);
    auto infer_request = compiled_model.create_infer_request();
    infer_request.infer();
}
