// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mem_usage.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "openvino/core/except.hpp"

#if defined _WIN32
#    include <windows.h>
#    include <dbghelp.h>
#    include <psapi.h>
#    include <shlwapi.h>
#    include <tlhelp32.h>
#else
#    include <dirent.h>
#    include <dlfcn.h>
#    include <execinfo.h>
#    include <signal.h>
#    include <sys/syscall.h>
#    include <sys/types.h>
#    include <unistd.h>

#    include <regex>
#endif

namespace intel_npu {
namespace memory_tracking {
namespace {

struct TrackerState {
    std::atomic<bool> tracking_active{false};
    int sampling_interval_ms = 1;

    std::thread tracking_thread;
    std::mutex events_mutex;
    std::mutex samples_mutex;

    std::vector<MemorySampleData> samples;

    std::chrono::steady_clock::time_point start_steady = std::chrono::steady_clock::now();
    std::chrono::system_clock::time_point start_system = std::chrono::system_clock::now();

    int64_t prev_working_set_bytes = 0;
    int64_t prev_private_working_set_bytes = 0;
    int64_t prev_peak_working_set_bytes = 0;
    bool has_previous_sample = false;
};

TrackerState& state() {
    static TrackerState s;
    return s;
}

std::string escape_json(const std::string& in) {
    std::ostringstream out;
    for (const char c : in) {
        switch (c) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << c;
            break;
        }
    }
    return out.str();
}

std::string derive_memory_signal(const TrackerState& s,
                                 const int64_t working_set_bytes,
                                 const int64_t peak_working_set_bytes,
                                 const int64_t private_working_set_bytes) {
    if (!s.has_previous_sample) {
        return "sample_start";
    }

    const int64_t ws_delta = working_set_bytes - s.prev_working_set_bytes;
    const int64_t pws_delta = private_working_set_bytes - s.prev_private_working_set_bytes;
    const int64_t peak_delta = peak_working_set_bytes - s.prev_peak_working_set_bytes;

    if (pws_delta > 0) {
        return "physical_resident_growth";
    }
    if (ws_delta > 0) {
        return "paged_or_mapped_growth";
    }
    if (peak_delta > 0) {
        return "high_watermark_growth";
    }
    if (pws_delta < 0) {
        return "physical_resident_drop";
    }
    if (ws_delta < 0) {
        return "paged_or_mapped_drop";
    }
    return "stable";
}

#if defined _WIN32

std::string basename_from_path(const std::string& path) {
    const size_t pos = path.find_last_of("\\/");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

using SymInitializeFn = BOOL(WINAPI*)(HANDLE, PCSTR, BOOL);
using SymSetOptionsFn = DWORD(WINAPI*)(DWORD);
using SymFromAddrFn = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);
using SymGetLineFromAddr64Fn = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD, PIMAGEHLP_LINE64);
using SymFunctionTableAccess64Fn = PVOID(WINAPI*)(HANDLE, DWORD64);
using SymGetModuleBase64Fn = DWORD64(WINAPI*)(HANDLE, DWORD64);
using StackWalk64Fn = BOOL(WINAPI*)(DWORD,
                                    HANDLE,
                                    HANDLE,
                                    LPSTACKFRAME64,
                                    PVOID,
                                    PREAD_PROCESS_MEMORY_ROUTINE64,
                                    PFUNCTION_TABLE_ACCESS_ROUTINE64,
                                    PGET_MODULE_BASE_ROUTINE64,
                                    PTRANSLATE_ADDRESS_ROUTINE64);

struct DbgHelpState {
    std::once_flag init_once;
    std::mutex api_mutex;
    HMODULE dbghelp_module = nullptr;
    SymInitializeFn sym_initialize = nullptr;
    SymSetOptionsFn sym_set_options = nullptr;
    SymFromAddrFn sym_from_addr = nullptr;
    SymGetLineFromAddr64Fn sym_get_line_from_addr64 = nullptr;
    SymFunctionTableAccess64Fn sym_function_table_access64 = nullptr;
    SymGetModuleBase64Fn sym_get_module_base64 = nullptr;
    StackWalk64Fn stack_walk64 = nullptr;
    bool ready = false;
};

DbgHelpState& dbghelp_state() {
    static DbgHelpState s;
    return s;
}

std::string to_lower_ascii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

const auto& tracked_module_order() {
    static const std::array modules = {"openvino_intel_npu_compiler_loader.dll",
                                       "openvino_intel_npu_compiler.dll",
                                       "npu_level_zero_umd.dll",
                                       "openvino_intel_npu_plugin.dll",
                                       "openvino.dll"};
    return modules;
}

void init_dbghelp_if_needed() {
    DbgHelpState& s = dbghelp_state();
    std::call_once(s.init_once, [&]() {
        s.dbghelp_module = LoadLibraryA("dbghelp.dll");
        if (s.dbghelp_module == nullptr) {
            return;
        }

        s.sym_initialize = reinterpret_cast<SymInitializeFn>(GetProcAddress(s.dbghelp_module, "SymInitialize"));
        s.sym_set_options = reinterpret_cast<SymSetOptionsFn>(GetProcAddress(s.dbghelp_module, "SymSetOptions"));
        s.sym_from_addr = reinterpret_cast<SymFromAddrFn>(GetProcAddress(s.dbghelp_module, "SymFromAddr"));
        s.sym_get_line_from_addr64 =
            reinterpret_cast<SymGetLineFromAddr64Fn>(GetProcAddress(s.dbghelp_module, "SymGetLineFromAddr64"));
        s.sym_function_table_access64 =
            reinterpret_cast<SymFunctionTableAccess64Fn>(GetProcAddress(s.dbghelp_module, "SymFunctionTableAccess64"));
        s.sym_get_module_base64 =
            reinterpret_cast<SymGetModuleBase64Fn>(GetProcAddress(s.dbghelp_module, "SymGetModuleBase64"));
        s.stack_walk64 = reinterpret_cast<StackWalk64Fn>(GetProcAddress(s.dbghelp_module, "StackWalk64"));

        if (s.sym_initialize == nullptr || s.sym_set_options == nullptr || s.sym_from_addr == nullptr ||
            s.sym_function_table_access64 == nullptr || s.sym_get_module_base64 == nullptr ||
            s.stack_walk64 == nullptr) {
            return;
        }

        constexpr DWORD options = SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES;
        s.sym_set_options(options);

        if (s.sym_initialize(GetCurrentProcess(), nullptr, TRUE) == FALSE) {
            return;
        }

        s.ready = true;
    });
}

std::string symbol_from_address(const uintptr_t addr) {
    init_dbghelp_if_needed();

    DbgHelpState& s = dbghelp_state();
    OPENVINO_ASSERT(s.ready, "DbgHelp is not ready");
    std::lock_guard<std::mutex> lock(s.api_mutex);

    std::array<unsigned char, sizeof(SYMBOL_INFO) + MAX_SYM_NAME> sym_storage{};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(sym_storage.data());
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = MAX_SYM_NAME;

    HMODULE module_handle = nullptr;
    std::string module_name;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(addr),
                           &module_handle) != FALSE &&
        module_handle != nullptr) {
        char module_path[MAX_PATH] = {0};
        const DWORD n = GetModuleFileNameA(module_handle, module_path, static_cast<DWORD>(MAX_PATH));
        if (n > 0) {
            module_name = basename_from_path(module_path);
        }
    }
    OPENVINO_ASSERT(!module_name.empty(), "Failed to get module name from address ", std::hex, addr);
    DWORD dwDisplacement = 0;
    std::ostringstream out;
    out << module_name << "!";
    if (s.sym_from_addr(GetCurrentProcess(),
                        static_cast<DWORD64>(addr),
                        reinterpret_cast<PDWORD64>(&dwDisplacement),
                        sym) == TRUE) {
        OPENVINO_ASSERT(std::string_view(sym->Name).find("std::") != 0,
                        "Symbol name begins with a function from standard namespace which are redundant ",
                        std::hex,
                        addr);
        out << sym->Name;
    } else {
        out << "0x" << std::hex << addr;
    }

    return out.str();
}

std::string join_stack_frames(const std::vector<std::string>& frames) {
    std::ostringstream oss;
    for (size_t i = 0; i < frames.size(); ++i) {
        if (i != 0) {
            oss << std::endl;
        }
        oss << frames[i];
    }
    return oss.str();
}

std::string capture_stack_trace_for_thread(HANDLE thread_handle, CONTEXT& context) {
    const size_t max_frames = 64;
    init_dbghelp_if_needed();
    DbgHelpState& dbg = dbghelp_state();
    if (!dbg.ready) {
        return "";
    }

    std::vector<std::string> frames;
    frames.reserve(max_frames);

    bool module_of_interest_found = false;

    STACKFRAME64 frame;
    std::memset(&frame, 0, sizeof(frame));

#    if defined(_M_X64)
    DWORD machine_type = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset = context.Rip;
    frame.AddrStack.Offset = context.Rsp;
    frame.AddrFrame.Offset = context.Rbp;
#    elif defined(_M_IX86)
    DWORD machine_type = IMAGE_FILE_MACHINE_I386;
    frame.AddrPC.Offset = context.Eip;
    frame.AddrStack.Offset = context.Esp;
    frame.AddrFrame.Offset = context.Ebp;
#    else
    return "";
#    endif
    frame.AddrPC.Mode = AddrModeFlat;
    frame.AddrStack.Mode = AddrModeFlat;
    frame.AddrFrame.Mode = AddrModeFlat;

    for (size_t i = 0; i < max_frames; ++i) {
        BOOL ok = FALSE;
        {
            struct ThreadSuspendResume {
                ThreadSuspendResume(HANDLE thread) : thread_handle(thread) {
                    SuspendThread(thread_handle);
                }
                ~ThreadSuspendResume() {
                    ResumeThread(thread_handle);
                }
                HANDLE thread_handle;
            } thread_suspend_resume(thread_handle);

            std::lock_guard<std::mutex> lock(dbg.api_mutex);
            ok = dbg.stack_walk64(machine_type,
                                  GetCurrentProcess(),
                                  thread_handle,
                                  &frame,
                                  &context,
                                  nullptr,
                                  reinterpret_cast<PFUNCTION_TABLE_ACCESS_ROUTINE64>(dbg.sym_function_table_access64),
                                  reinterpret_cast<PGET_MODULE_BASE_ROUTINE64>(dbg.sym_get_module_base64),
                                  nullptr);

            if (ok == FALSE || frame.AddrPC.Offset == 0) {
                break;
            }
        }

        try {
            auto symbol = symbol_from_address(frame.AddrPC.Offset);
            module_of_interest_found =
                module_of_interest_found ||
                std::any_of(tracked_module_order().begin(), tracked_module_order().end(), [&](const char* mod) {
                    return symbol.find(mod) != std::string::npos;
                });
            frames.push_back(symbol);
        } catch (...) {
            DWORD errorMessageID = ::GetLastError();
            if (errorMessageID != 0) {
                LPSTR messageBuffer = nullptr;
                size_t size = FormatMessageA(
                    FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                    NULL,
                    errorMessageID,
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    (LPSTR)&messageBuffer,
                    0,
                    NULL);
                if (messageBuffer) {
                    LocalFree(messageBuffer);
                }
            }
        }
    }
    if (!module_of_interest_found) {
        return "";
    }
    return join_stack_frames(frames);
}

void collect_sample_windows(MemorySampleData& sample) {
    TrackerState& s = state();
    init_dbghelp_if_needed();

    const auto now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - s.start_steady);

    // Capture memory counters for this sample
    PROCESS_MEMORY_COUNTERS_EX mem_counters;
    std::memset(&mem_counters, 0, sizeof(mem_counters));
    mem_counters.cb = sizeof(mem_counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&mem_counters),
                              sizeof(mem_counters))) {
        return;
    }

    const int64_t working_set_bytes = static_cast<int64_t>(mem_counters.WorkingSetSize);
    const int64_t peak_working_set_bytes = static_cast<int64_t>(mem_counters.PeakWorkingSetSize);
    const int64_t private_working_set_bytes = static_cast<int64_t>(mem_counters.PrivateUsage);
    const bool sample_changed = !s.has_previous_sample || working_set_bytes != s.prev_working_set_bytes ||
                                private_working_set_bytes != s.prev_private_working_set_bytes ||
                                peak_working_set_bytes != s.prev_peak_working_set_bytes;

    // Query current PID for all threads and iterate through them
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        return;
    }

    const DWORD pid = GetCurrentProcessId();
    const DWORD self_tid = GetCurrentThreadId();
    std::vector<std::pair<uint32_t, std::string>> raw_stacks_per_thread;

    THREADENTRY32 te;
    std::memset(&te, 0, sizeof(te));
    te.dwSize = sizeof(te);

    if (!sample.is_instrumentation && Thread32First(snap, &te) != FALSE) {
        do {
            if (te.th32OwnerProcessID != pid || te.th32ThreadID == self_tid) {
                continue;
            }

            HANDLE thread_handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                                              FALSE,
                                              te.th32ThreadID);
            if (thread_handle == nullptr) {
                continue;
            }

            // Collect snapshot from this specific thread
            CONTEXT context{};
            context.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(thread_handle, &context) != FALSE) {
                const std::string stack = capture_stack_trace_for_thread(thread_handle, context);
                if (!stack.empty()) {
                    raw_stacks_per_thread.push_back(
                        std::make_pair(static_cast<uint32_t>(GetThreadId(thread_handle)), stack));
                }
            }

            CloseHandle(thread_handle);
        } while (Thread32Next(snap, &te) != FALSE);
    }
    CloseHandle(snap);

    const std::string memory_signal =
        derive_memory_signal(s, working_set_bytes, peak_working_set_bytes, private_working_set_bytes);

    // Store the sample record
    if ((sample_changed && !raw_stacks_per_thread.empty()) || sample.is_instrumentation) {
        sample.timestamp = now_ns;
        sample.working_set_bytes = working_set_bytes;
        sample.peak_working_set_bytes = peak_working_set_bytes;
        sample.private_working_set_bytes = private_working_set_bytes;
        sample.raw_stacks_per_threads = raw_stacks_per_thread;
        sample.memory_signal = memory_signal;
        std::lock_guard<std::mutex> lk(s.samples_mutex);
        s.samples.push_back(sample);
    }

    s.prev_working_set_bytes = working_set_bytes;
    s.prev_private_working_set_bytes = private_working_set_bytes;
    s.prev_peak_working_set_bytes = peak_working_set_bytes;
    s.has_previous_sample = true;
}

#else

struct LinuxStackCapture {
    static constexpr int max_frames = 64;
    std::atomic<bool> ready{false};
    void* frames[max_frames]{};
    int frame_count = 0;
};

LinuxStackCapture g_linux_stack_capture;
std::mutex g_linux_stack_capture_mutex;

void linux_stack_signal_handler(int /*signo*/) {
    g_linux_stack_capture.frame_count = backtrace(g_linux_stack_capture.frames, LinuxStackCapture::max_frames);
    g_linux_stack_capture.ready.store(true, std::memory_order_release);
}

void install_linux_stack_signal_handler() {
    static std::once_flag once;
    std::call_once(once, []() {
        struct sigaction sa{};
        sa.sa_handler = linux_stack_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGRTMIN + 3, &sa, nullptr);
    });
}

std::string resolve_address_linux(void* addr) {
    Dl_info info;
    if (dladdr(addr, &info) != 0) {
        std::ostringstream oss;
        if (info.dli_fname != nullptr) {
            const char* sep = std::strrchr(info.dli_fname, '/');
            oss << (sep ? sep + 1 : info.dli_fname);
        } else {
            oss << "unknown";
        }
        oss << "!";
        if (info.dli_sname != nullptr) {
            oss << info.dli_sname;
        } else {
            oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(addr);
        }
        return oss.str();
    }
    std::ostringstream oss;
    oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(addr);
    return oss.str();
}

std::string capture_stack_trace_linux_for_tid(pid_t tid) {
    std::lock_guard<std::mutex> lock(g_linux_stack_capture_mutex);
    g_linux_stack_capture.ready.store(false, std::memory_order_release);
    g_linux_stack_capture.frame_count = 0;

    if (syscall(SYS_tgkill, getpid(), static_cast<long>(tid), SIGRTMIN + 3) != 0) {
        return "";
    }

    constexpr int max_wait_us = 10000;
    int waited_us = 0;
    while (!g_linux_stack_capture.ready.load(std::memory_order_acquire) && waited_us < max_wait_us) {
        usleep(100);
        waited_us += 100;
    }

    if (!g_linux_stack_capture.ready.load(std::memory_order_acquire)) {
        return "";
    }

    std::ostringstream oss;
    for (int i = 0; i < g_linux_stack_capture.frame_count; ++i) {
        if (i != 0) {
            oss << "\n";
        }
        oss << resolve_address_linux(g_linux_stack_capture.frames[i]);
    }
    return oss.str();
}

std::vector<pid_t> enumerate_thread_tids() {
    std::vector<pid_t> tids;
    DIR* task_dir = opendir("/proc/self/task");
    if (task_dir == nullptr) {
        return tids;
    }
    struct dirent* entry;
    while ((entry = readdir(task_dir)) != nullptr) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        const pid_t tid = static_cast<pid_t>(std::atoi(entry->d_name));
        if (tid > 0) {
            tids.push_back(tid);
        }
    }
    closedir(task_dir);
    return tids;
}

void collect_sample_linux(MemorySampleData& sample) {
    TrackerState& s = state();
    install_linux_stack_signal_handler();

    const auto now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - s.start_steady);

    int64_t vm_rss_kb = 0;
    int64_t vm_hwm_kb = 0;

    std::ifstream status_file("/proc/self/status");
    std::string line;
    std::smatch vm_match;
    std::regex vm_rss_regex("VmRSS:");
    std::regex vm_hwm_regex("VmHWM:");

    while (std::getline(status_file, line)) {
        if (std::regex_search(line, vm_match, vm_rss_regex)) {
            std::istringstream iss(vm_match.suffix());
            iss >> vm_rss_kb;
        } else if (std::regex_search(line, vm_match, vm_hwm_regex)) {
            std::istringstream iss(vm_match.suffix());
            iss >> vm_hwm_kb;
        }
    }

    const int64_t working_set_bytes = vm_rss_kb * 1024;
    const int64_t peak_working_set_bytes = vm_hwm_kb * 1024;
    const int64_t private_working_set_bytes = vm_rss_kb * 1024;

    const bool sample_changed = !s.has_previous_sample || working_set_bytes != s.prev_working_set_bytes ||
                                private_working_set_bytes != s.prev_private_working_set_bytes ||
                                peak_working_set_bytes != s.prev_peak_working_set_bytes;

    const pid_t self_tid = static_cast<pid_t>(syscall(SYS_gettid));
    std::vector<std::pair<uint32_t, std::string>> raw_stacks_per_thread;

    if (!sample.is_instrumentation) {
        const std::vector<pid_t> tids = enumerate_thread_tids();
        for (const pid_t tid : tids) {
            if (tid == self_tid) {
                continue;
            }
            const std::string stack = capture_stack_trace_linux_for_tid(tid);
            if (!stack.empty()) {
                raw_stacks_per_thread.push_back(std::make_pair(static_cast<uint32_t>(tid), stack));
            }
        }
    }

    const std::string memory_signal =
        derive_memory_signal(s, working_set_bytes, peak_working_set_bytes, private_working_set_bytes);

    if ((sample_changed && !raw_stacks_per_thread.empty()) || sample.is_instrumentation) {
        sample.timestamp = now_ns;
        sample.working_set_bytes = working_set_bytes;
        sample.peak_working_set_bytes = peak_working_set_bytes;
        sample.private_working_set_bytes = private_working_set_bytes;
        sample.raw_stacks_per_threads = raw_stacks_per_thread;
        sample.memory_signal = memory_signal;
        std::lock_guard<std::mutex> lock(s.samples_mutex);
        s.samples.push_back(sample);
    }

    s.prev_working_set_bytes = working_set_bytes;
    s.prev_private_working_set_bytes = private_working_set_bytes;
    s.prev_peak_working_set_bytes = peak_working_set_bytes;
    s.has_previous_sample = true;
}

#endif

void sampling_worker_loop() {
    TrackerState& s = state();
    while (s.tracking_active.load()) {
        MemorySampleData sample;
#if defined _WIN32
        collect_sample_windows(sample);
#else
        collect_sample_linux(sample);
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(s.sampling_interval_ms));
    }
}

std::string to_ts_us(const std::chrono::nanoseconds ns) {
    return std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(ns).count());
}

std::string perfetto_default_path_json() {
#if defined _WIN32
    return "npu_memory_trace.perfetto.json";
#else
    return "npu_memory_trace.perfetto.json";
#endif
}

std::string perfetto_default_path_csv() {
#if defined _WIN32
    return "NPUTrace_PerformanceCounter.csv";
#else
    return "NPUTrace_PerformanceCounter.csv";
#endif
}

struct AutoTrackerLifecycle {
    AutoTrackerLifecycle() {
        MemoryTracker::instance().start_tracking(1);
    }

    ~AutoTrackerLifecycle() {
        auto& tracker = MemoryTracker::instance();
        tracker.stop_tracking();

        const char* json_path = std::getenv("NPU_MEMORY_TRACE_JSON");
        const char* csv_path = std::getenv("NPU_MEMORY_TRACE_CSV");

        if (json_path != nullptr && json_path[0] != '\0') {
            tracker.export_perfetto_trace(json_path);
        } else {
            tracker.export_perfetto_trace(perfetto_default_path_json());
        }
        if (csv_path != nullptr && csv_path[0] != '\0') {
            tracker.export_performance_monitor_csv(csv_path);
        } else {
            tracker.export_performance_monitor_csv(perfetto_default_path_csv());
        }
    }
};

AutoTrackerLifecycle g_auto_tracker_lifecycle;

}  // namespace

MemoryTracker& MemoryTracker::instance() {
    static MemoryTracker tracker;
    return tracker;
}

MemoryTracker::MemoryTracker() = default;

MemoryTracker::~MemoryTracker() {
    stop_tracking();
}

void MemoryTracker::start_tracking(const int sampling_interval_ms) {
    TrackerState& s = state();
    if (s.tracking_active.load()) {
        return;
    }

    s.sampling_interval_ms = (sampling_interval_ms > 0) ? sampling_interval_ms : 100;
    s.start_steady = std::chrono::steady_clock::now();
    s.start_system = std::chrono::system_clock::now();

    {
        std::lock_guard<std::mutex> sm_lock(s.samples_mutex);
        s.samples.clear();
    }

    s.prev_working_set_bytes = 0;
    s.prev_private_working_set_bytes = 0;
    s.prev_peak_working_set_bytes = 0;
    s.has_previous_sample = false;

    s.tracking_active.store(true);
    s.tracking_thread = std::thread(&sampling_worker_loop);
}

void MemoryTracker::stop_tracking() {
    TrackerState& s = state();
    if (!s.tracking_active.load()) {
        return;
    }

    s.tracking_active.store(false);
    if (s.tracking_thread.joinable()) {
        s.tracking_thread.join();
    }
}

void MemoryTracker::export_perfetto_trace(const std::string& output_file) {
    TrackerState& s = state();

    std::vector<MemorySampleData> samples_copy;
    {
        std::lock_guard<std::mutex> lock(s.samples_mutex);
        samples_copy = s.samples;
    }

    struct OrderedItem {
        std::chrono::nanoseconds ts;
        bool operator<(const OrderedItem& other) const {
            return ts < other.ts;
        }
        size_t idx;
    };

    std::vector<OrderedItem> ordered;
    for (size_t i = 0; i < samples_copy.size(); ++i) {
        ordered.push_back({samples_copy[i].timestamp, i});
    }
    std::sort(ordered.begin(), ordered.end(), [](const OrderedItem& a, const OrderedItem& b) {
        return a.ts < b.ts;
    });

    constexpr uint32_t ws_tid = 2;
    constexpr uint32_t pws_tid = 3;
    constexpr uint32_t peak_tid = 4;
    constexpr uint32_t events_tid = 5;
    constexpr uint32_t instrumentation_tid = 6;

    const std::string path = output_file.empty() ? perfetto_default_path_json() : output_file;
    std::ofstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Cannot open file for Perfetto trace export: " + path);
    }

    file << "[";
    std::chrono::nanoseconds max_ts{};

    auto write_meta = [&](const char* name, const char* value, const uint32_t tid) {
        file << "\n{";
        file << "\"name\":\"" << name << "\",";
        file << "\"ph\":\"M\",";
#if defined _WIN32
        file << "\"pid\":" << GetCurrentProcessId() << ",";
#else
        file << "\"pid\":" << getpid() << ",";
#endif
        file << "\"tid\":" << tid << ",";
        file << "\"args\":{";
        file << "\"name\":\"" << value << "\"";
        file << "}},\n";
    };

#ifdef _WIN32
    char exePath[MAX_PATH];
    DWORD size = MAX_PATH;
    QueryFullProcessImageNameA(GetCurrentProcess(), 0, exePath, &size);
    write_meta("process_name", PathFindFileNameA(exePath), static_cast<uint32_t>(GetCurrentProcessId()));
#elif defined(__linux__)
    std::ifstream commFile("/proc/self/comm");
    std::string process_name;
    if (commFile.is_open()) {
        std::getline(commFile, process_name);
    }
    write_meta("process_name", process_name.c_str(), getpid());
#endif
    write_meta("thread_name", "memory.events", events_tid);
    write_meta("thread_name", "npu.functions", instrumentation_tid);

    for (const auto& item : ordered) {
        const auto& sample = samples_copy[item.idx];
        if (sample.timestamp > max_ts) {
            max_ts = sample.timestamp;
        }

        auto write_counter = [&](const char* name, const double value, const uint32_t tid) {
            file << "\n{";
            file << "\"name\":\"" << name << "\",";
            file << "\"ph\":\"C\",";
            file << "\"ts\":" << to_ts_us(sample.timestamp) << ",";
#if defined _WIN32
            file << "\"pid\":" << GetCurrentProcessId() << ",";
#else
            file << "\"pid\":" << getpid() << ",";
#endif
            file << "\"tid\":" << tid << ",";
            file << "\"args\":{";
            file << "\"value_mb\":" << value;
            file << "}},\n";
        };

        write_counter("memory.working_set", sample.working_set_bytes / (1024.0 * 1024.0), ws_tid);
        write_counter("memory.private_working_set", sample.private_working_set_bytes / (1024.0 * 1024.0), pws_tid);
        write_counter("memory.peak_working_set", sample.peak_working_set_bytes / (1024.0 * 1024.0), peak_tid);

        auto write_event =
            [&](const char* name, const std::vector<std::pair<uint32_t, std::string>>& stack, const uint32_t tid) {
                file << "\n{";
                file << "\"name\":\"" << name << "\",";
                file << "\"ts\":" << to_ts_us(sample.timestamp) << ",";
                if (!sample.is_instrumentation) {
                    file << "\"ph\":\"X\",";
                    file << "\"dur\":"
                         << to_ts_us(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::milliseconds(state().sampling_interval_ms)))
                         << ",";
                } else {
                    if (sample.is_begin) {
                        file << "\"ph\":\"B\",";
                    } else {
                        file << "\"ph\":\"E\",";
                    }
                }
#if defined _WIN32
                file << "\"pid\":" << GetCurrentProcessId() << ",";
#else
                file << "\"pid\":" << getpid() << ",";
#endif
                file << "\"tid\":" << tid << ",";
                file << "\"args\":{";
                if (!stack.empty()) {
                    file << "\"stack\":\"";
                    for (const auto& frame : stack) {
                        file << "\\nthread_id=" << frame.first << "\\n";
                        file << escape_json(frame.second) << "\\n";
                    }
                    file << "\"";
                }
                file << "}},\n";
            };

        if (sample.is_instrumentation) {
            write_event(sample.function.c_str(), {}, instrumentation_tid);
        } else {
            write_event(sample.memory_signal.c_str(), sample.raw_stacks_per_threads, events_tid);
        }
    }

    // Remove trailing comma if needed
    file.seekp(-3, std::ios_base::end);
    file << "\n]\n";
}

void MemoryTracker::export_performance_monitor_csv(const std::string& output_file) {
    TrackerState& s = state();

    std::vector<MemorySampleData> samples_copy;
    {
        std::lock_guard<std::mutex> lock(s.samples_mutex);
        samples_copy = s.samples;
    }

    const std::string path = output_file.empty() ? perfetto_default_path_csv() : output_file;
    std::ofstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Cannot open file for Performance Monitor CSV export: " + path);
    }

#if defined _WIN32
    TIME_ZONE_INFORMATION tzi;
    DWORD tz_res = GetTimeZoneInformation(&tzi);
    (void)tz_res;

    const int offset_minutes = -static_cast<int>(tzi.Bias);

    file << "\"(PDH-CSV 4.0) (Local Time)(" << offset_minutes << ")\",";
    file << "\"\\\\NPUTrace\\Process V2(NPUMemoryTracer:" << GetCurrentProcessId() << ")\\Working Set\",";
    file << "\"\\\\NPUTrace\\Process V2(NPUMemoryTracer:" << GetCurrentProcessId() << ")\\Working Set - Private\",";
    file << "\"\\\\NPUTrace\\Process V2(NPUMemoryTracer:" << GetCurrentProcessId() << ")\\Peak Working Set\",";
    file << "\"Use this template to create a basic Data Collector Set.\"\n";
#else
    file << "\"(PDH-CSV 4.0) (Local Time)(0)\",";
    file << "\"\\\\NPUTrace\\Process V2(NPUMemoryTracer:" << getpid() << ")\\Working Set\",";
    file << "\"\\\\NPUTrace\\Process V2(NPUMemoryTracer:" << getpid() << ")\\Working Set - Private\",";
    file << "\"\\\\NPUTrace\\Process V2(NPUMemoryTracer:" << getpid() << ")\\Peak Working Set\",";
    file << "\"Use this template to create a basic Data Collector Set.\"\n";
#endif

    for (const auto& sample : samples_copy) {
        const auto absolute_tp =
            s.start_system + std::chrono::duration_cast<std::chrono::system_clock::duration>(sample.timestamp);
        const std::time_t t = std::chrono::system_clock::to_time_t(absolute_tp);

        std::tm tm_val{};
#if defined _WIN32
        localtime_s(&tm_val, &t);
#else
        localtime_r(&t, &tm_val);
#endif

        char date_buf[64];
        std::strftime(date_buf, sizeof(date_buf), "%m/%d/%Y %H:%M:%S", &tm_val);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(sample.timestamp).count() % 1000;

        file << "\"" << date_buf << "." << std::setfill('0') << std::setw(3) << ms << "\",";
        file << "\"" << sample.working_set_bytes << "\",";
        file << "\"" << sample.private_working_set_bytes << "\",";
        file << "\"" << sample.peak_working_set_bytes << "\",";
    }
}

MemoryTrackerHelper::MemoryTrackerHelper(const char* function) : _function(function) {
    MemorySampleData sample_data;
    sample_data.is_instrumentation = true;
    sample_data.is_begin = true;
    sample_data.function = _function;
#ifdef _WIN32
    collect_sample_windows(sample_data);
#else
    collect_sample_linux(sample_data);
#endif
}

MemoryTrackerHelper::~MemoryTrackerHelper() {
    MemorySampleData sample_data;
    sample_data.is_instrumentation = true;
    sample_data.is_begin = false;
    sample_data.function = _function;
#ifdef _WIN32
    collect_sample_windows(sample_data);
#else
    collect_sample_linux(sample_data);
#endif
}

}  // namespace memory_tracking

int64_t get_peak_memory_usage() {
#if defined _WIN32
    PROCESS_MEMORY_COUNTERS mem_counters;
    std::memset(&mem_counters, 0, sizeof(mem_counters));

    if (!GetProcessMemoryInfo(GetCurrentProcess(), &mem_counters, sizeof(mem_counters))) {
        OPENVINO_THROW("Can't get system memory values");
    }

    static constexpr double bytes_in_kilobyte = 1024.0;
    return static_cast<int64_t>(std::round(mem_counters.PeakWorkingSetSize / bytes_in_kilobyte));
#else
    std::size_t peak_mem_usage_kb = 0;

    std::ifstream status_file("/proc/self/status");
    std::string line;
    std::regex vm_peak_regex("VmPeak:");
    std::smatch vm_match;
    bool mem_values_found = false;

    while (std::getline(status_file, line)) {
        if (std::regex_search(line, vm_match, vm_peak_regex)) {
            std::istringstream iss(vm_match.suffix());
            iss >> peak_mem_usage_kb;
            mem_values_found = true;
            break;
        }
    }

    if (!mem_values_found) {
        OPENVINO_THROW("Can't get system memory values");
    }

    return static_cast<int64_t>(peak_mem_usage_kb);
#endif
}

}  // namespace intel_npu
