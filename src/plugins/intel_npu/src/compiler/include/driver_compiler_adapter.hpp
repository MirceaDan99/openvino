// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {
namespace driverCompilerAdapter {

/**
 * @brief Adapter for Compiler in driver
 * @details Wrap compiler in driver calls and do preliminary actions (like opset conversion)
 */
class LevelZeroCompilerAdapter final : public ICompiler {
public:
    LevelZeroCompilerAdapter();

    uint32_t getSupportedOpsetVersion() const override final;

    std::shared_ptr<NetworkDescription> compile(const std::shared_ptr<const ov::Model>& model,
                               const Config& config) const override final;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override final;

    NetworkMetadata parse(const uint8_t* blobData, size_t blobSize, const Config& config) const override final;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const uint8_t* blobData,
                                                            size_t blobSize,
                                                            const Config& config) const override final;

private:
    /**
     * @brief Separate externals calls to separate class
     */
    std::shared_ptr<ICompiler> apiAdapter;
    ze_driver_handle_t _driverHandle = nullptr;
    Logger _logger;
};

}  // namespace driverCompilerAdapter
}  // namespace intel_npu
