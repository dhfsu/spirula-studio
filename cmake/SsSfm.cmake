# The Structure-from-Motion module (src/sfm/), a Vulkan-only
# subsystem that replaces the COLMAP subprocess. See src/sfm/README.md.
#
# Defines:
#   ss_sfm          static library (SfM pipeline + embedded SPIR-V)
#   sfm_*_test      one executable per src/sfm/tests/*.cpp
#
# The `spirula-sfm` CLI is defined by the standalone app module.
#
# This module needs Vulkan and slangc but NOT the compute backend: it carries
# its own Vulkan context (src/sfm/vk/) and its own SPIR-V blobs, so it builds
# as a standalone Vulkan module.

include(SsVulkan)
ss_vulkan_lib()
find_package(Threads REQUIRED)

set(SS_SFM_SRC ${SS_SRC}/sfm)
set(SS_SFM_SHADERS ${SS_SFM_SRC}/shaders)

# ---------------------------------------------------------------------------
# Shader variant matrix
# ---------------------------------------------------------------------------
# One BA blob per cached (Real, Loss) pair (src/sfm/README.md). slangc aborts
# on every ba_df_* variant under Windows (docs/build.md), so no df default there.
if(WIN32)
    set(_sfm_reals_default "float;double")
else()
    set(_sfm_reals_default "float;double;df")
endif()
set(SS_SFM_REALS "${_sfm_reals_default}" CACHE STRING
    "SfM bundle-adjustment scalar configurations to compile")
set(SS_SFM_LOSSES "trivial;huber;cauchy" CACHE STRING
    "SfM bundle-adjustment robust losses to compile")

include(SsSlang)
ss_find_slangc(SS_SFM_SLANGC)
ss_build_spirv_tool(SS_SFM_SPIRV_TOOL)

set(_sfm_spirv_dir ${CMAKE_CURRENT_BINARY_DIR}/sfm_spirv)
set(_sfm_embed_cpp ${CMAKE_CURRENT_BINARY_DIR}/sfm_shaders_embedded.cpp)

# -I<shaders> lets a shader reach the shared device math by its shaders-relative
# path (`#include "common/camera.slang"`), matching the C++ convention.
set(_sfm_slang_args -target spirv -O2 -fvk-use-entrypoint-name
    -I${SS_SFM_SHADERS})

file(GLOB _sfm_ba_deps CONFIGURE_DEPENDS
    ${SS_SFM_SHADERS}/ba/*.slang ${SS_SFM_SHADERS}/common/*.slang)

set(_sfm_blobs "")
foreach(real ${SS_SFM_REALS})
    if(real STREQUAL "float")
        set(_rdef -DREAL_FLOAT)
    elseif(real STREQUAL "double")
        set(_rdef -DREAL_DOUBLE)
    elseif(real STREQUAL "df")
        set(_rdef -DREAL_DF)
    else()
        message(FATAL_ERROR "unknown SS_SFM_REALS entry '${real}' "
            "(expected float, double or df)")
    endif()
    foreach(loss ${SS_SFM_LOSSES})
        if(loss STREQUAL "trivial")
            set(_ldef "")
        elseif(loss STREQUAL "huber")
            set(_ldef -DLOSS_HUBER)
        elseif(loss STREQUAL "cauchy")
            set(_ldef -DLOSS_CAUCHY)
        else()
            message(FATAL_ERROR "unknown SS_SFM_LOSSES entry '${loss}' "
                "(expected trivial, huber or cauchy)")
        endif()

        set(_name ba_${real}_${loss})
        set(_out ${_sfm_spirv_dir}/${_name}.spv)
        if(real STREQUAL "df")
            # Compile to .raw.spv, then decorate. slangc emits no NoContraction
            # and some drivers then contract float expressions, destroying the
            # error-free transforms the emulated double-float type is built on.
            set(_raw ${_sfm_spirv_dir}/${_name}.raw.spv)
            add_custom_command(OUTPUT ${_out}
                COMMAND ${SS_SFM_SLANGC} ${SS_SFM_SHADERS}/ba/ba.slang
                        -o ${_raw} ${_sfm_slang_args} ${_rdef} ${_ldef}
                COMMAND ${SS_SFM_SPIRV_TOOL} nocontract ${_raw} ${_out}
                DEPENDS ${_sfm_ba_deps} ${SS_SFM_SPIRV_TOOL}
                COMMENT "SPIR-V ${_name} (+NoContraction)"
                VERBATIM)
        else()
            add_custom_command(OUTPUT ${_out}
                COMMAND ${SS_SFM_SLANGC} ${SS_SFM_SHADERS}/ba/ba.slang
                        -o ${_out} ${_sfm_slang_args} ${_rdef} ${_ldef}
                DEPENDS ${_sfm_ba_deps}
                COMMENT "SPIR-V ${_name}"
                VERBATIM)
        endif()
        list(APPEND _sfm_blobs ${_out})
    endforeach()
endforeach()

# Single-blob stages: blob name, source, the directory the dependency glob
# watches, comma-separated defines. The matcher is built four times: with and
# without the integer dot product, at 128- and 256-byte descriptors.
foreach(stage sift:sift/sift.slang:sift:none
              match:match/bruteforce.slang:match:none
              match_nodot:match/bruteforce.slang:match:-DNO_DOT4
              match_d256:match/bruteforce.slang:match:-DDESC_WORDS=64
              match_nodot_d256:match/bruteforce.slang:match:-DNO_DOT4,-DDESC_WORDS=64)
    string(REPLACE ":" ";" _parts ${stage})
    list(GET _parts 0 _name)
    list(GET _parts 1 _rel)
    list(GET _parts 2 _dir)
    list(GET _parts 3 _def)
    if(_def STREQUAL "none")
        set(_def "")
    else()
        string(REPLACE "," ";" _def "${_def}")
    endif()
    file(GLOB _deps CONFIGURE_DEPENDS
        ${SS_SFM_SHADERS}/${_dir}/*.slang ${SS_SFM_SHADERS}/common/*.slang)
    set(_out ${_sfm_spirv_dir}/${_name}.spv)
    add_custom_command(OUTPUT ${_out}
        COMMAND ${SS_SFM_SLANGC} ${SS_SFM_SHADERS}/${_rel}
                -o ${_out} ${_sfm_slang_args} ${_def}
        DEPENDS ${_deps}
        COMMENT "SPIR-V ${_name}"
        VERBATIM)
    list(APPEND _sfm_blobs ${_out})
endforeach()

list(LENGTH _sfm_blobs _sfm_nblobs)
message(STATUS "SfM SPIR-V: ${_sfm_nblobs} blobs "
    "(reals: ${SS_SFM_REALS}; losses: ${SS_SFM_LOSSES})")

# List file for the embed step (keeps the command line short).
set(_sfm_listfile ${_sfm_spirv_dir}/blobs.txt)
string(REPLACE ";" "\n" _sfm_listbody "${_sfm_blobs}")
ss_write_if_different(${_sfm_listfile} "${_sfm_listbody}\n")

add_custom_command(OUTPUT ${_sfm_embed_cpp}
    COMMAND ${SS_SFM_SPIRV_TOOL} embed --sfm ${_sfm_embed_cpp}
            --list ${_sfm_listfile}
    DEPENDS ${SS_SFM_SPIRV_TOOL} ${_sfm_blobs} ${_sfm_listfile}
    COMMENT "Embedding SfM SPIR-V (${_sfm_nblobs} blobs)"
    VERBATIM)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SS_SFM_SOURCES CONFIGURE_DEPENDS ${SS_SFM_SRC}/*.cpp)
list(FILTER SS_SFM_SOURCES EXCLUDE REGEX "/tests/")
list(FILTER SS_SFM_SOURCES EXCLUDE REGEX "/vk/spirv_tool\.cpp$")

add_library(ss_sfm STATIC
    ${SS_SFM_SOURCES}
    ${_sfm_embed_cpp}
    # Image and compression implementations are part of this standalone library.
    ${SS_SRC}/external/stb_image_impl.cpp
    ${SS_SRC}/external/stb_image_write_impl.cpp
    ${SS_SRC}/core/ExrImage.cpp
    ${SS_SRC}/core/SceneAlign.cpp
    ${SS_SRC}/external/miniz.c
)
target_include_directories(ss_sfm PUBLIC ${SS_SRC})
target_link_libraries(ss_sfm PUBLIC ss_vulkan Threads::Threads ss_i18n)

# The learned frontends are disabled in this standalone build; their option
# names remain parseable and report a clear runtime error.
target_compile_definitions(ss_sfm PUBLIC SS_HAVE_ALIKED=0 SS_HAVE_LOMA=0)
target_compile_options(ss_sfm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${SPLAT_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:C>:${SPLAT_C_FLAGS}>)
set_property(TARGET ss_sfm PROPERTY CXX_STANDARD 17)

# Most of the pipeline is header-only; list headers so IDEs and Ninja see them.
file(GLOB_RECURSE SS_SFM_HEADERS CONFIGURE_DEPENDS ${SS_SFM_SRC}/*.h)
target_sources(ss_sfm PRIVATE ${SS_SFM_HEADERS})

# ---------------------------------------------------------------------------
# Tests -- one executable per file.
# ---------------------------------------------------------------------------
file(GLOB SS_SFM_TESTS CONFIGURE_DEPENDS ${SS_SFM_SRC}/tests/*.cpp)
foreach(test_src ${SS_SFM_TESTS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE ss_sfm)
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 17)
    target_compile_options(${test_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${SPLAT_CXX_FLAGS}>)
endforeach()
