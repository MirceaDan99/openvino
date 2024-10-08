# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(BUILD_SHARED_LIBS OFF)


# We have to avoid linking against loader with debug postfix due it's a part of driver
# and the name will be the same for release and debug configurations
set(CMAKE_DEBUG_POSTFIX "")

# Skip warnings as errors for thirdparty
set(ze_loader_sources_flags) # list, not string

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(ze_loader_sources_flags /WX-)
    # solve pdb access issue
    set(USE_Z7 ON)
    # Close spectre for ze loader
    set(ze_loader_sources_flags ${ze_loader_sources_flags} /Qspectre-)
elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    set(ze_loader_sources_flags -Wno-error -Wno-undef -Wno-missing-declarations)
    if(UNUSED_BUT_SET_VARIABLE_SUPPORTED)
        set(ze_loader_sources_flags ${ze_loader_sources_flags} -Wno-unused-but-set-variable)
    endif()
endif()
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
add_subdirectory(level-zero EXCLUDE_FROM_ALL)

# Level-zero thirdparty treats warnings as errors internally, need to set compile options for its source files
get_target_property(ze_loader_sources ze_loader SOURCES)
list(FILTER ze_loader_sources INCLUDE REGEX ".*.cpp")
set_source_files_properties(${ze_loader_sources} TARGET_DIRECTORY ze_loader PROPERTIES COMPILE_OPTIONS "${ze_loader_sources_flags}")

set_property(TARGET ze_loader APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/level-zero/include>)

# This VERSION file created by L0 may cause compilation issue of oneTBB headers, so remove it
file(REMOVE "${CMAKE_BINARY_DIR}/VERSION")
