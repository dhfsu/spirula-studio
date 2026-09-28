cmake_minimum_required(VERSION 3.18)
project(spirula_sfm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(SS_SRC ${SS_ROOT}/src)
set(SS_VERSION "2026.9.24")
set(SS_DEFAULT_LANG "en" CACHE STRING "Default interface language")
set(SPLAT_CXX_FLAGS -O3 -Wno-sign-compare)
set(SPLAT_C_FLAGS -O3)

list(APPEND CMAKE_MODULE_PATH ${SS_ROOT}/cmake)

function(ss_write_if_different path content)
    if(EXISTS "${path}")
        file(READ "${path}" _existing)
        if(_existing STREQUAL "${content}")
            return()
        endif()
    endif()
    file(WRITE "${path}" "${content}")
endfunction()

include(SsVulkan)
include(SsSlang)
find_package(Threads REQUIRED)

add_library(ss_i18n STATIC src/i18n/Locale.cpp)
target_include_directories(ss_i18n PUBLIC ${SS_SRC})
target_compile_definitions(ss_i18n PUBLIC SS_DEFAULT_LANG=${SS_DEFAULT_LANG})

include(SsSfm)
include(SsSfmApp)

message(STATUS "spirula-sfm configured")
