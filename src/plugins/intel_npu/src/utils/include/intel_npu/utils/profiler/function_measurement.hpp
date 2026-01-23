// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>

#include "openvino/util/file_util.hpp"

#define OV_NPU_ENABLE_PROFILER 0
#if OV_NPU_ENABLE_PROFILER
constexpr std::string_view output_file_path = "ov_npu_profiler.csv";

namespace intel_npu {

namespace utils {

class Timer {
public:
    static std::shared_ptr<Timer>& getInstance() {
        static auto timerPtr = std::make_shared<Timer>();
        return timerPtr;
    }

    void record(const std::string_view& func_name) {
        if (func_name.empty()) {
            return;
        }
        auto it = records.find(func_name.data());
        if (it == records.end()) {
            records.emplace(std::make_pair(func_name, std::make_pair(.0, 0u)));
        }
    }

    void record(const std::string_view& func_name, const std::string_view& append_func_name, const std::chrono::high_resolution_clock::time_point& start) {
        auto now = std::chrono::high_resolution_clock::now();
        if (func_name.empty()) {
            return;
        }
        decltype(records)::iterator it;
        if (!append_func_name.empty()) {
            std::string new_func_name = std::string(func_name.data()) + "-" + append_func_name.data();
            it = records.find(new_func_name);
            if (it == records.end()) {
                it = records.find(func_name.data());
                OPENVINO_ASSERT(it != records.end(), "At least old func name should be found!");
                auto nh = records.extract(it->first);
                nh.key() = new_func_name;
                records.insert(std::move(nh));
                it = records.find(new_func_name);
            } else {
                records.erase(func_name.data());
            }
        } else {
            it = records.find(func_name.data());
        }
        OPENVINO_ASSERT(it != records.end(), "Record should be added at this point");
        it->second.first += std::chrono::duration_cast<std::chrono::microseconds>(
                                    now - start).count() / 1000.0;
        it->second.second++;
    }

    ~Timer() {
        std::string tmp_output_file_path = output_file_path.data();
        uint64_t unique_id = 1;
        while (ov::util::file_exists(tmp_output_file_path)) {
            tmp_output_file_path = output_file_path.data();
            std::string new_suffix = "_" + std::to_string(unique_id++) + ".csv";
            tmp_output_file_path.replace(tmp_output_file_path.find(".csv"), std::string::npos, new_suffix);
        }
        std::ofstream file_out(tmp_output_file_path);
        file_out << "Function,Times called,Time per call (ms),Total time (ms)" << std::endl;
        for (const auto& pair : records) {
            file_out << std::fixed << pair.first << "," << pair.second.second << ","
                     << pair.second.first / pair.second.second << ","
                     << pair.second.first << std::endl;
            /* file_out << pair.first << "," << std::get<2>(pair.second) << ","
                << std::get<1>(pair.second) << "," << std::get<1>(pair.second) * std::get<2>(pair.second) << std::endl; */
        }
        file_out.close();
    }

private:
    std::map<
        std::string,
        std::pair</* total_time_ms */ double, /* times_called */ uint64_t>>
        records;
};

class TimerHelper {
public:
    TimerHelper() = delete;

    TimerHelper(const TimerHelper& other) = delete;

    TimerHelper(TimerHelper&& other) {
        _current_func_name = other._current_func_name;
        _append_func_name = other._append_func_name;
        other._current_func_name = "";
        other._append_func_name = "";
        start = std::chrono::high_resolution_clock::now();
    }

    TimerHelper(const std::string_view& func_name) : _current_func_name{func_name}, _append_func_name{} {
        Timer::getInstance()->record(func_name);
        start = std::chrono::high_resolution_clock::now();
    }

    ~TimerHelper() {
        Timer::getInstance()->record(_current_func_name, _append_func_name, start);
    }

    void append_func_name(const std::string_view& append_func_name_) {
        this->_append_func_name = append_func_name_;
    }

    TimerHelper& operator=(const TimerHelper& other) = delete;

    TimerHelper& operator=(TimerHelper&& other) = delete; /* {
        _current_func_name = std::move(other._current_func_name);
        _append_func_name = std::move(other._append_func_name);
        start = std::chrono::high_resolution_clock::now();
        return *this;
    } */

private:
    std::string_view _current_func_name, _append_func_name;
    std::chrono::high_resolution_clock::time_point start;
};

}  // namespace utils

}  // namespace intel_npu

#    define OV_NPU_PROFILE_FUNCTION() auto timer = intel_npu::utils::TimerHelper(__FUNCTION__); std::vector<intel_npu::utils::TimerHelper> _timers;
#    define STRINGIFY(X)     #X
#    define _STRINGIFIED(X)  STRINGIFY(X)
#    define STRINGIFIED(X)   _STRINGIFIED(X)
// #    define _MAKE_TIMER_VAR(X)   timer_##STRINGIFY_LINE(__LINE__)
// #    define MAKE_TIMER_VAR(X)    _MAKE_TIMER_VAR(X)
#    define OV_NPU_PROFILE_FUNCTION_LINE()                                                                 \
        /* _Pragma("warning(push)") */                                                                     \
        /* _Pragma("warning(disable : 4456 )") */ /* supress redeclaration warning */                      \
        _timers.push_back(intel_npu::utils::TimerHelper(__FUNCTION__"#L"STRINGIFIED(__LINE__)));
#    define OV_NPU_PROFILE_FUNCTION_LINE_END()                                                             \
        /* _Pragma("warning(pop)") */                                                                      \
        _timers.back().append_func_name(STRINGIFIED(__LINE__));                                            \
        _timers.pop_back();
// #    undef MAKE_TIMER_VAR
// #    undef _MAKE_TIMER_VAR
#else
#    define OV_NPU_PROFILE_FUNCTION()
#    define OV_NPU_PROFILE_FUNCTION_LINE()
#    define OV_NPU_PROFILE_FUNCTION_LINE_END()
#endif
#undef OV_NPU_NEABLE_PROFILER
