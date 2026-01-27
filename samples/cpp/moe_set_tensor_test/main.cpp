// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

/**
 * @brief Simple test application to reproduce set_tensor performance issue
 * 
 * This application:
 * 1. Loads IR model from specified path
 * 2. Compiles model for NPU device
 * 3. Creates random L0 (Level Zero) tensors
 * 4. Repeatedly calls set_tensor() and infer()
 * 5. Reports performance statistics
 */

// Timer helper class
class Timer {
public:
    Timer() : total_time_(0), count_(0) {}

    template<typename Func>
    void record(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        total_time_ += duration;
        count_++;
    }

    double total_ms() const { return total_time_ / 1000.0; }
    double avg_ms() const { return count_ > 0 ? total_ms() / count_ : 0.0; }
    size_t count() const { return count_; }

private:
    long long total_time_;  // microseconds
    size_t count_;
};

// Generate random nf4 data
void generate_random_nf4_data(int8_t* data, size_t num_elements) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 15);  // nf4 has 16 values (0-15)

    for (size_t i = 0; i < num_elements; ++i) {
        int8_t value = static_cast<int8_t>(dis(gen));  // Randomly select index 0-15
        if (i % 2 == 0) {
            data[i / 2] = (value & 0x0F);  // Store lower 4 bits
        } else {
            data[i / 2] |= (value << 4);  // Store upper 4 bits
        }
    }
}

// Create L0 tensor with random data
ov::RemoteTensor create_random_l0_tensor(ov::RemoteContext& remote_context,
                                         const ov::element::Type& type,
                                         const ov::Shape& shape) {
    // Calculate tensor size
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }
    size_t byte_size = num_elements * type.size();

    // Create remote tensor
    ov::RemoteTensor remote_tensor = remote_context.create_tensor(type, shape);

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
        host_tensor.copy_to(remote_tensor);
        return remote_tensor;
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
    host_tensor.copy_to(remote_tensor);

    return remote_tensor;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        std::string model_path = "Model0_kv1152_02_REP0108_moe_chunk_0.blob";
        std::string device_name = "NPU";
        size_t num_iterations = 10000;

        if (argc > 1) {
            model_path = argv[1];
        }
        if (argc > 2) {
            device_name = argv[2];
        }
        if (argc > 3) {
            num_iterations = std::stoi(argv[3]);
        }

        std::cout << "========================================" << std::endl;
        std::cout << "MoE set_tensor Performance Test" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Model path: " << model_path << std::endl;
        std::cout << "Device: " << device_name << std::endl;
        std::cout << "Iterations: " << num_iterations << std::endl;
        std::cout << "========================================" << std::endl;

        // Step 1: Initialize OpenVINO Core
        std::cout << "\n[Step 1] Initializing OpenVINO Core..." << std::endl;
        ov::Core core;

        // Step 2: Get remote context for Level Zero
        std::cout << "[Step 2] Creating remote context for NPU..." << std::endl;
        ov::RemoteContext remote_context = core.get_default_context(device_name);
        std::cout << "  Remote context created" << std::endl;

        // Step 3: Import compiled model from blob
        std::cout << "[Step 3] Importing compiled model from: " << model_path << std::endl;
        std::ifstream blob_file(model_path, std::ios::binary);
        if (!blob_file.is_open()) {
            throw std::runtime_error("Failed to open blob file: " + model_path);
        }
        ov::CompiledModel compiled_model = core.import_model(blob_file, remote_context);
        blob_file.close();
        std::cout << "  Model imported successfully" << std::endl;
        std::cout << "  Inputs: " << compiled_model.inputs().size() << std::endl;
        std::cout << "  Outputs: " << compiled_model.outputs().size() << std::endl;

        // Print input information
        for (size_t i = 0; i < compiled_model.inputs().size(); ++i) {
            const auto& input = compiled_model.inputs()[i];
            std::cout << "    Input[" << i << "]: " 
                      << input.get_any_name() << " "
                      << input.get_element_type() << " "
                      << input.get_shape() << std::endl;
        }

        // Step 4: Create infer request
        std::cout << "[Step 4] Creating infer request..." << std::endl;
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        std::cout << "  Infer request created" << std::endl;

        // Step 5: Create a pool of random L0 tensors for all inputs
        // This simulates MoE scenario where different expert weights are used each iteration
        const size_t tensor_pool_size = 30;  // Number of different tensors per input
        std::cout << "[Step 5] Creating tensor pool (" << tensor_pool_size << " tensors per input)..." << std::endl;
        std::vector<std::vector<ov::RemoteTensor>> input_tensor_pools;
        
        for (size_t i = 0; i < compiled_model.inputs().size(); ++i) {
            const auto& input = compiled_model.inputs()[i];
            std::cout << "  Creating tensor pool for input[" << i << "]: " 
                      << input.get_any_name() << std::endl;
            
            std::vector<ov::RemoteTensor> tensor_pool;
            for (size_t j = 0; j < tensor_pool_size; ++j) {
                auto remote_tensor = create_random_l0_tensor(
                    remote_context,
                    input.get_element_type(),
                    input.get_shape()
                );
                tensor_pool.push_back(remote_tensor);
            }
            input_tensor_pools.push_back(tensor_pool);
        }
        std::cout << "  All tensor pools created" << std::endl;

        // Step 6: Warmup run
        /* std::cout << "[Step 6] Running warmup..." << std::endl;
        for (size_t i = 0; i < compiled_model.inputs().size(); ++i) {
            infer_request.set_tensor(compiled_model.inputs()[i], input_tensor_pools[i][0]);
        }
        infer_request.infer();
        std::cout << "  Warmup completed" << std::endl; */

        // Step 7: Performance test loop with random tensor selection
        std::cout << "\n[Step 7] Running performance test (" << num_iterations << " iterations)..." << std::endl;
        std::cout << "  Note: Each iteration randomly selects tensors from pool to simulate MoE expert switching" << std::endl;
        std::getchar();
        Timer set_tensor_timer;
        Timer infer_timer;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, tensor_pool_size - 1);
        size_t tensor_idx = 1;

        for (size_t iter = 0; iter < num_iterations; ++iter) {
            // Measure set_tensor time with randomly selected tensors
            // set_tensor_timer.record([&]() {
                for (size_t i = 0; i < compiled_model.inputs().size(); ++i) {
                    // Randomly select a tensor from the pool for this input
                    // size_t tensor_idx = dis(gen);
                    infer_request.set_tensor(compiled_model.inputs()[i], input_tensor_pools[i][tensor_idx]);
                    tensor_idx = ++tensor_idx % 30;
                }
            // });

            // Measure infer time
            // infer_timer.record([&]() {
                infer_request.infer();
            // });

            // Print progress every 10 iterations
            /* if ((iter + 1) % 10 == 0) {
                std::cout << "  Progress: " << (iter + 1) << "/" << num_iterations << std::endl;
            } */
        }

        // Step 8: Print statistics
        /* std::cout << "\n========================================" << std::endl;
        std::cout << "Performance Statistics" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "set_tensor():" << std::endl;
        std::cout << "  Count:      " << set_tensor_timer.count() << std::endl;
        std::cout << "  Total time: " << set_tensor_timer.total_ms() << " ms" << std::endl;
        std::cout << "  Avg time:   " << set_tensor_timer.avg_ms() << " ms" << std::endl;
        std::cout << std::endl;
        std::cout << "infer():" << std::endl;
        std::cout << "  Count:      " << infer_timer.count() << std::endl;
        std::cout << "  Total time: " << infer_timer.total_ms() << " ms" << std::endl;
        std::cout << "  Avg time:   " << infer_timer.avg_ms() << " ms" << std::endl;
        std::cout << std::endl;
        std::cout << "set_tensor/infer ratio: " 
                  << (set_tensor_timer.avg_ms() / infer_timer.avg_ms() * 100.0) << "%" << std::endl;
        std::cout << "========================================" << std::endl; */

        return EXIT_SUCCESS;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
