// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"
// clang-format on

#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"

/**
 * @brief Print OV Parameters
 * @param reference on OV Parameter
 * @return void
 */
void print_any_value(const ov::Any& value) {
    if (value.empty()) {
        slog::info << "EMPTY VALUE" << slog::endl;
    } else {
        std::string stringValue = value.as<std::string>();
        slog::info << (stringValue.empty() ? "\"\"" : stringValue) << slog::endl;
    }
}

ov::Tensor read_or_write_reference(const std::string_view ref_path,
                                   const ov::Tensor& ref_tensor,
                                   const bool forceDelete = false) {
    ov::Tensor output_ref_tensor;
    std::fstream ref_file(ref_path.data(), std::ios::in | std::ios::binary);
    OPENVINO_ASSERT(ref_file || forceDelete, "Could not open reference file '", ref_path, "' for reading!");
    if (forceDelete) {
        ref_file.close();
        ref_file.open(ref_path.data(), std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
        ref_file.write(reinterpret_cast<const char*>(ref_tensor.data()), ref_tensor.get_byte_size());
        ref_file.close();
        ref_file.open(ref_path.data(), std::ios::in | std::ios::binary);
    }
    ref_file.seekg(0, std::ios::end);
    output_ref_tensor = ov::Tensor(ov::element::u8, ov::Shape{static_cast<uint64_t>(ref_file.tellg())});
    ref_file.seekg(0, std::ios::beg);
    ref_file.read(reinterpret_cast<char*>(output_ref_tensor.data()), output_ref_tensor.get_byte_size());
    ref_file.close();
    return output_ref_tensor;
}

void generate_input_tensors(const ov::CompiledModel& compiled_model, ov::InferRequest& infer_request) {
    for (const auto& input : compiled_model.inputs()) {
        /* const */ auto& tensor = infer_request.get_tensor(input);
        // ov::Tensor new_tensor(tensor.get_element_type(), tensor.get_shape());
        //  auto new_tensor = context.create_host_tensor(tensor.get_element_type(), tensor.get_shape());
        std::string tensor_str;
        tensor_str.resize(tensor.get_byte_size());
        // std::string_view tensor_str(static_cast<const char*>(tensor.data()), tensor.get_byte_size());
        std::iota(tensor_str.begin(), tensor_str.end(), static_cast<unsigned char>(0));
        std::memcpy(tensor.data(), tensor_str.c_str(), tensor.get_byte_size());
        //  std::memcpy(new_tensor.data(), tensor_str.c_str(), new_tensor.get_byte_size());
        // std::memset(tensor.data(), 5, tensor.get_byte_size());
        // std::memset(new_tensor.data(), 5, new_tensor.get_byte_size());
        // infer_request.set_tensor(input, new_tensor);
    }
}

ov::Tensor generate_big_output_tensor(const ov::CompiledModel& compiled_model, ov::InferRequest& infer_request) {
    size_t totalOutputSize = 0;
    for (const auto& output : compiled_model.outputs()) {
        totalOutputSize += infer_request.get_tensor(output).get_byte_size();
    }
    ov::Tensor output_tensor(ov::element::u8, ov::Shape{totalOutputSize});
    totalOutputSize = 0;
    for (const auto& output : compiled_model.outputs()) {
        const auto& tensor = infer_request.get_tensor(output);
        std::memcpy(static_cast<char*>(output_tensor.data()) + totalOutputSize,
                    static_cast<const char*>(tensor.data()),
                    tensor.get_byte_size());
        totalOutputSize += tensor.get_byte_size();
    }
    return output_tensor;
}

void compare_actual_vs_ref_tensor(const ov::Tensor& output_tensor, const ov::Tensor& output_ref_tensor) {
    OPENVINO_ASSERT(output_tensor.get_byte_size() == output_ref_tensor.get_byte_size(),
                    "Size mismatch between the actual vs reference tensors");
    for (size_t i = 0; i < output_ref_tensor.get_byte_size(); ++i) {
        const char actual_val = static_cast<const char*>(output_tensor.data())[i];
        const char ref_val = static_cast<const char*>(output_ref_tensor.data())[i];
        OPENVINO_ASSERT(actual_val == ref_val,
                        "Mismatch detected at index ",
                        i,
                        " actual_val = ",
                        (int)actual_val,
                        " ref_val = ",
                        (int)ref_val);
    }
}

bool is_npu_target_device(const char* target_device) {
    return std::string_view(target_device) == "NPU";
}

int main(int argc, char* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc < 3) {
            std::cout << "Usage : " << argv[0] << " <model_path> "
                      << "<compiler_type> <platform> <weights_path> (optional)-IMPORT_FIRST DEVICE "
                         "(optional)-DISABLE_WEIGHTLESS"
                      << std::endl;
            return EXIT_FAILURE;
        }

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Get list of available devices --------
        std::vector<std::string> availableDevices = core.get_available_devices();

        // -------- Step 3. Query and print supported metrics and config keys --------
        slog::info << "Available devices: " << slog::endl;
        for (auto&& device : availableDevices) {
            slog::info << device << slog::endl;

            // Query supported properties and print all of them
            slog::info << "\tSUPPORTED_PROPERTIES: " << slog::endl;
            auto supported_properties = core.get_property(device, ov::supported_properties);
            for (auto&& property : supported_properties) {
                if (property != ov::supported_properties.name()) {
                    slog::info << "\t\t" << (property.is_mutable() ? "Mutable: " : "Immutable: ") << property << " : "
                               << slog::flush;
                    print_any_value(core.get_property(device, property));
                }
            }

            slog::info << slog::endl;
        }

        auto target_device = argc > 6 ? argv[6] : "NPU";
        ov::AnyMap config = is_npu_target_device(target_device) ? ov::AnyMap{ 
                                                /* ov::cache_dir(std::string("C:\\Work\\mirceaau") + std::string_view(argv[1]).substr(std::string_view(argv[1]).find_last_of('\\')).data() + ".bin"), */
                                                ov::intel_npu::bypass_umd_caching(true),
                                                {ov::intel_npu::compiler_type.name(), argv[2]},
                                                (argc > 7 && std::string_view(argv[7]) == "DISABLE_WEIGHTLESS") ? ov::enable_weightless(false) : ov::enable_weightless(true),
                                                {ov::intel_npu::platform.name(), argv[3]},
                                                /* ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE) */
                                                /* ov::internal::model_sharing_context(modelSharingContext) */ } : ov::AnyMap{
                                                    (argc > 7 && std::string_view(argv[7]) == "DISABLE_WEIGHTLESS") ? ov::enable_weightless(false) : ov::enable_weightless(true),
                                                };
        if (argc > 4) {
            if (std::string_view(argv[4]) != "MODEL_PTR") {
                config.emplace(ov::weights_path(argv[4]));
            } else {
                config.emplace(ov::hint::model(core.read_model(argv[1])));
            }
        }

        auto blob_path = std::string("C:\\Work\\mirceaau") +
                         std::string_view(argv[1]).substr(std::string_view(argv[1]).find_last_of('\\')).data() + "." +
                         target_device + ".blob";
        auto ref_path = std::string("C:\\Work\\mirceaau") +
                        std::string_view(argv[1]).substr(std::string_view(argv[1]).find_last_of('\\')).data() + "." +
                        target_device + ".outputs_ref.bin";

        if (argc > 5 && std::string_view(argv[5]) == "IMPORT_FIRST") {
            std::ifstream blob_stream(blob_path, std::ios::in | std::ios::binary);
            if (blob_stream) {
                auto compiled_model = core.import_model(blob_stream, target_device, config);
                auto infer_request = compiled_model.create_infer_request();
                {
                    generate_input_tensors(compiled_model, infer_request);
                    infer_request.infer();
                    auto output_tensor = generate_big_output_tensor(compiled_model, infer_request);
                    auto output_ref_tensor =
                        read_or_write_reference(ref_path, output_tensor, /* forceDelete = */ false);
                    compare_actual_vs_ref_tensor(output_tensor, output_ref_tensor);
                }
                infer_request = {};
                compiled_model = {};
                config.erase(ov::hint::model.name());
                blob_stream.close();
            }
        }

        auto model = core.read_model(argv[1]);
        std::map<ov::Output<ov::Node>, ov::PartialShape> partial_shapes;
        auto params = model->get_parameters();
        std::cout << "Parameters:" << std::endl;
        for (const auto& param : params) {
            auto partial_shape = param->get_output_partial_shape(0);
            auto name = param->outputs()[0].get_names().empty() ? param->get_friendly_name()
                                                                : param->outputs()[0].get_any_name();
            std::cout << "\t - " << name << ":" << param->get_element_type() << partial_shape.to_string() << std::endl;
            ;
            if (partial_shape.is_dynamic()) {
                for (auto& dimension : partial_shape) {
                    if (dimension.get_interval().get_min_val() != dimension.get_interval().get_max_val()) {
                        dimension = ov::Dimension(1, 1);
                    }
                }
                partial_shapes[param->get_default_output()] = partial_shape;
            }
        }
        if (!partial_shapes.empty() && is_npu_target_device(target_device)) {
            model->reshape(partial_shapes);
        }

        std::cout << "Results:" << std::endl;
        for (const auto& result : model->get_results()) {
            auto partial_shape = result->get_output_partial_shape(0);
            auto name = result->outputs()[0].get_names().empty() ? result->get_friendly_name()
                                                                 : result->outputs()[0].get_any_name();
            std::cout << "\t - " << name << ":" << result->get_element_type() << partial_shape.to_string() << std::endl;
            ;
        }

        auto compiled_model = core.compile_model(model, target_device, config);
        auto infer_request = compiled_model.create_infer_request();
        {
            generate_input_tensors(compiled_model, infer_request);
            infer_request.infer();
            auto output_tensor = generate_big_output_tensor(compiled_model, infer_request);

            auto output_ref_tensor = read_or_write_reference(ref_path, output_tensor, /* forceDelete = */ true);
            compare_actual_vs_ref_tensor(output_tensor, output_ref_tensor);
        }
        infer_request = {};

        FILE* file = fopen(blob_path.c_str(), "wb+");
        {
            std::fstream ss(file);
            if (!ss.is_open()) {
                throw std::runtime_error("Couldn't open file stream!");
            }
            compiled_model.export_model(ss);
            ss.flush();
            ss.seekg(0, std::ios::beg);
            compiled_model = {};
            model = {};

            auto compiled_model = core.import_model(ss, target_device, config);
            auto infer_request = compiled_model.create_infer_request();
            {
                generate_input_tensors(compiled_model, infer_request);
                infer_request.infer();
                auto output_tensor = generate_big_output_tensor(compiled_model, infer_request);
                auto output_ref_tensor = read_or_write_reference(ref_path, output_tensor, /* forceDelete = */ false);
                compare_actual_vs_ref_tensor(output_tensor, output_ref_tensor);
            }
            infer_request = {};
            compiled_model = {};

            core.unload_plugin(target_device);
            fclose(file);
        }
        // std::cout << "Stop before exit..."; getchar();

    } catch (const std::exception& ex) {
        std::cerr << std::endl << "Exception occurred: " << ex.what() << std::endl << std::flush;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
