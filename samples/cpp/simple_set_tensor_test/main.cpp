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

#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"

/**
 * @brief Simple test application to reproduce set_tensor performance issue
 * 
 * This application:
 * 1. Compiles simple 2 input subtract model from test utils
 * 2. Allocates L0 tensor via get_tensor()
 * 4. Repeatedly calls set_tensor() and infer()
 * 5. Reports performance statistics
 */

/**
 * ========================================
 * Performance Statistics
 * ========================================
 * set_tensor():
 *  Count:      1000
 *  Total time: 5.222 ms
 *  Avg time:   0.005 ms    # however, this was for 2 inputs model, for 29 inputs moe model it would give 0.170 ms / 29 ~= 0.0058 ms
 *
 * infer():
 *  Count:      1000
 *  Total time: 116.002 ms
 *  Avg time:   0.116 ms

 * set_tensor/infer ratio: 4.502%
========================================
**/

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

int main(int argc, char* argv[]) {
    try {
        std::string device_name = "NPU";
        size_t num_iterations = 1000;

        std::cout << "========================================" << std::endl;
        std::cout << "Simple set_tensor Performance Test" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Device: " << device_name << std::endl;
        std::cout << "Iterations: " << num_iterations << std::endl;
        std::cout << "========================================" << std::endl;

        // Step 1: Initialize OpenVINO Core
        std::cout << "\n[Step 1] Initializing OpenVINO Core..." << std::endl;
        ov::Core core;

        // Step 3: Compile simple 2 input susbstract model
        std::cout << "[Step 2] Compile simple 2 input susbstract model from test utils..." << std::endl;
        ov::CompiledModel compiled_model = core.compile_model(ov::test::utils::make_2_input_subtract(), "NPU");
        std::cout << "  Model compiled successfully" << std::endl;
        std::cout << "  Inputs: " << compiled_model.inputs().size() << std::endl;
        std::cout << "  Outputs: " << compiled_model.outputs().size() << std::endl;

        // Print input information
        for (size_t i = 0; i < compiled_model.inputs().size(); ++i) {
            const auto& input = compiled_model.inputs()[i];
            std::cout << "    Input[" << i << "]: " 
                      // << input.get_any_name() << " "
                      << input.get_element_type() << " "
                      << input.get_shape() << std::endl;
        }

        // Step 4: Create infer request
        std::cout << "[Step 3] Creating infer request..." << std::endl;
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        std::cout << "  Infer request created" << std::endl;

        // Step 5: get_tensor
        std::cout << "[Step 4] Allocate tensor..." << std::endl;
        auto tensors = std::vector<ov::Tensor>{infer_request.get_tensor(compiled_model.inputs()[0]), infer_request.get_tensor(compiled_model.inputs()[1])};
        std::memset(static_cast<void*>(tensors.at(0).data()), 0, tensors.at(0).get_byte_size());
        std::memset(static_cast<void*>(tensors.at(1).data()), 0, tensors.at(1).get_byte_size());

        // Step 6: Warmup run
        std::cout << "[Step 5] Running warmup..." << std::endl;
        infer_request.infer();
        std::cout << "  Warmup completed" << std::endl;

        // Step 7: Performance test loop with random tensor selection
        std::cout << "\n[Step 6] Running performance test (" << num_iterations << " iterations)..." << std::endl;
        std::cout << "  Note: Each iteration randomly selects tensors from pool to simulate MoE expert switching" << std::endl;
        // std::getchar();
        Timer set_tensor_timer;
        Timer infer_timer;

        for (size_t iter = 0; iter < num_iterations; ++iter) {
            // Measure set_tensor time with randomly selected tensors
            set_tensor_timer.record([&]() {
                for (size_t i = 0; i < compiled_model.inputs().size(); ++i) {
                    // Randomly select a tensor from the pool for this input
                    infer_request.set_tensor(compiled_model.inputs()[i], tensors.at(iter % 2 == 1 ? i : (i + 1) % compiled_model.inputs().size()));
                }
            });

            // Measure infer time
            infer_timer.record([&]() {
                infer_request.infer();
            });

            // Print progress every 10 iterations
            if ((iter + 1) % 10 == 0) {
                std::cout << "  Progress: " << (iter + 1) << "/" << num_iterations << std::endl;
            }
        }

        // Step 8: Print statistics
        std::cout << "\n========================================" << std::endl;
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
        std::cout << "========================================" << std::endl;

        return EXIT_SUCCESS;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
