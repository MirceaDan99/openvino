// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace intel_npu {

/// @brief Get peak memory usage in kilobytes
int64_t get_peak_memory_usage();

namespace memory_tracking {

struct MemorySampleData {
    std::chrono::nanoseconds timestamp{};
    int64_t working_set_bytes = 0;
    int64_t peak_working_set_bytes = 0;
    int64_t private_working_set_bytes = 0;
    std::vector<std::pair<uint32_t, std::string>> raw_stacks_per_threads = {};
    std::string memory_signal;
    bool is_instrumentation = false;
    bool is_begin = false;
    std::string function;
};

class MemoryTrackerHelper {
public:
    MemoryTrackerHelper(const char* function);
    ~MemoryTrackerHelper();
    MemoryTrackerHelper(const MemoryTrackerHelper&) = delete;
    MemoryTrackerHelper& operator=(const MemoryTrackerHelper&) = delete;

private:
    const char* _function;
};

/// @brief Memory tracker singleton
class MemoryTracker {
public:
    /// @brief Get the singleton instance
    static MemoryTracker& instance();

    /// @brief Start memory tracking with periodic sampling
    /// @param sampling_interval_ms Interval between memory samples in milliseconds (default: 100ms)
    void start_tracking(int sampling_interval_ms = 100);

    /// @brief Stop memory tracking
    void stop_tracking();

    /// @brief Export memory trace in Perfetto JSON format (compatible with ui.perfetto.dev)
    /// @param output_file Path to output JSON file
    void export_perfetto_trace(const std::string& output_file);

    /// @brief Export memory samples in Windows Performance Monitor CSV format
    /// @param output_file Path to output CSV file
    /// @note CSV will use generic "NPUTrace" computer name instead of actual hostname
    /// @note Timezone information is automatically detected and included
    void export_performance_monitor_csv(const std::string& output_file);

    MemoryTracker(const MemoryTracker&) = delete;
    MemoryTracker& operator=(const MemoryTracker&) = delete;

private:
    MemoryTracker();
    ~MemoryTracker();
};

}  // namespace memory_tracking

#define NPU_TRACE_MEMORY_EVENT() NPU_TRACE_MEMORY_EVENT_IMPL(__FUNCTION__)

#define NPU_TRACE_MEMORY_EVENT_IMPL(function) \
    intel_npu::memory_tracking::MemoryTrackerHelper _unused_memory_tracker_helper_instance(function);

}  // namespace intel_npu
