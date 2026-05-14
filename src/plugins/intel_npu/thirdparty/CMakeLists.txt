# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# zeroApi
#

if(ENABLE_NPU_PLUGIN_ENGINE)
    add_library(level-zero-ext INTERFACE)
    set_property(TARGET level-zero-ext APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/level-zero-ext/>)
    add_library(LevelZero::NPUExt ALIAS level-zero-ext)
endif()

#
# yaml-cpp
#

if(ENABLE_INTEL_NPU_PROTOPIPE)
    set(YAML_BUILD_SHARED_LIBS OFF)
    add_subdirectory(yaml-cpp EXCLUDE_FROM_ALL)
    # NB: Suppress warnings in yaml-cpp
    if(SUGGEST_OVERRIDE_SUPPORTED)
        target_compile_options(yaml-cpp PRIVATE -Wno-suggest-override)
    endif()
endif()

#
# tcmalloc
#

set(BUILD_TESTING OFF)  # Will create a gtest target that will be in conflict with the one added by OV
set(BUILD_SHARED_LIBS ON) # So far it is best to have executables linked to dynamic version of tcmalloc, WindowsInfo::Patch would throw if both executable and plugin dll link against same static version of tcmalloc.lib. Moreover, application must make sure that tcmalloc.dll is kept alive all the time, otherwise patched kernel32 calls would trigger deallocated functions 
set(gperftools_build_minimal OFF) # Minimal version won't include heap profiler
# set(GPERFTOOLS_BUILD_DEBUGALLOC ON) # Unfortunately Windows version for allocations debug is not yet implemented, could be useful for stack traces printing

# add_compile_options("/D PERFTOOLS_DLL_DECL=") # Only used within static tcmalloc.lib
# add_compile_definitions("TCMALLOC_SKIP_OVERRIDE") # Windows kernel32 heap allocation functions won't be patched if we activate this. However it also has an runtime equivalent using env var `set TCMALLOC_DISABLE_REPLACEMENT=1`
# add_compile_definitions("WIN32_OVERRIDE_ALLOCATORS;TCMALLOC_OVERRIDE_ONLY_CXX") # This would be needed only if we want to use the optimized versions of Thread-Caching mallocs (tcmalloc). However, heap profiler is implemented only within Windows patched kernel32 functions to track allocations/deallocations.

# Other useful env vars:
# `set TCMALLOC_DISABLE_REPLACEMENT=1` // also described above, do not patch windows kernel32 functions for allocations/deallocations (HeapProfileStart, HeapProfileDump and HeapProfileStop calls will not include any heap traces)
# it is best to deactivate all the other intermediary head profile dumpings, so far threading mechanism implemented in this library is quite unstable
# `set HEAP_PROFILE_ALLOCATION_INTERVAL=0`
# `set HEAP_PROFILE_DEALLOCATION_INTERVAL=0`
# `set HEAP_PROFILE_INUSE_INTERVAL=0`
# `set HEAP_PROFILE_TIME_INTERVAL=0`

add_subdirectory(gperftools)
