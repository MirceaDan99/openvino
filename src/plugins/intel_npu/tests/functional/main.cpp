// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>

#include <functional_test_utils/summary/op_summary.hpp>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "intel_npu/config/config.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "npu_test_report.hpp"
#include "npu_test_tool.hpp"

#ifdef _WIN32
#    include <dbghelp.h>
#    include <process.h>
#    include <windows.h>
#else
#    include <execinfo.h>
#endif

namespace testing {
namespace internal {
extern bool g_help_flag;
}  // namespace internal
}  // namespace testing

void sigsegv_handler(int errCode);

void sigsegv_handler(int errCode) {
    auto& s = ov::test::utils::OpSummary::getInstance();
    s.saveReport();
    std::cerr << "Unexpected application crash with code: " << errCode << std::endl;

#ifdef _WIN32
    unsigned int i;
    void* stack[100];
    unsigned short frames;
    SYMBOL_INFO* symbol;
    HANDLE process;

    process = GetCurrentProcess();

    SymInitialize(process, NULL, TRUE);

    frames = CaptureStackBackTrace(0, 100, stack, NULL);
    symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    for (i = 0; i < frames; i++) {
        SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
        std::cerr << frames - i - 1 << ": " << symbol->Name << std::hex << symbol->Address;
    }

    free(symbol);
#elif defined(__linux__)
    void* array[100];
    size_t size;

    // Get the stack trace (up to 10 addresses)
    size = backtrace(array, 100);

    // Print out the stack trace
    backtrace_symbols_fd(array, size, STDERR_FILENO);
#endif

    std::abort();
}

int main(int argc, char** argv, char** envp) {
    // register crashHandler for SIGSEGV signal
    signal(SIGSEGV, sigsegv_handler);
    signal(SIGABRT, sigsegv_handler);
    signal(SIGSEGV, sigsegv_handler);
    signal(SIGILL, sigsegv_handler);
#ifdef __linux__
    signal(SIGFPE, sigsegv_handler);
    signal(SIGBUS, sigsegv_handler);
#endif

    std::ostringstream oss;
    oss << "Command line args (" << argc << "): ";
    for (int c = 0; c < argc; ++c) {
        oss << " " << argv[c];
    }
    oss << std::endl;

#ifdef WIN32
    oss << "Process id: " << _getpid() << std::endl;
#else
    oss << "Process id: " << getpid() << std::endl;
#endif

    std::cout << oss.str();
    oss.str("");

    oss << "Environment variables: ";
    for (char** env = envp; *env != 0; env++) {
        oss << *env << "; ";
    }

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new ov::test::utils::NpuTestReportEnvironment());

    const bool dryRun = ::testing::GTEST_FLAG(list_tests) || ::testing::internal::g_help_flag;

    if (!dryRun) {
        const std::string noFetch{"<not fetched>"};
        std::string backend{noFetch}, arch{noFetch}, full{noFetch};
        try {
            ov::test::utils::NpuTestTool npuTestTool(ov::test::utils::NpuTestEnvConfig::getInstance());
            backend = npuTestTool.getDeviceMetric(ov::intel_npu::backend_name.name());
            arch = npuTestTool.getDeviceMetric(ov::device::architecture.name());
            full = npuTestTool.getDeviceMetric(ov::device::full_name.name());
        } catch (const std::exception& e) {
            std::cerr << "Exception while trying to determine device characteristics: " << e.what() << std::endl;
        }
        std::cout << "Tests run with: Backend name: '" << backend << "'; Device arch: '" << arch
                  << "'; Full device name: '" << full << "'" << std::endl;
    }

    std::string dTest = ::testing::internal::GTEST_FLAG(internal_run_death_test);
    if (dTest.empty()) {
        std::cout << oss.str() << std::endl;
    } else {
        std::cout << "gtest death test process is running" << std::endl;
    }

    auto& log = intel_npu::Logger::global();
    auto level = ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_LOG_LEVEL;
    ov::log::Level logLevel =
        level.empty() ? ov::log::Level::ERR : intel_npu::OptionParser<ov::log::Level>::parse(level.c_str());
    log.setLevel(logLevel);

    return RUN_ALL_TESTS();
}
