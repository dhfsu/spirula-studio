# SfM 模块，独立使用 Vulkan 上下文与嵌入式 SPIR-V；参见 src/sfm/README.md。
# 定义静态库 ss_sfm，并为 src/sfm/tests/*.cpp 分别生成 sfm_*_test。
# spirula-sfm CLI 由应用模块定义；本模块依赖 Vulkan 和 slangc，不依赖训练计算后端。

include(SsVulkan)
ss_vulkan_lib()
find_package(Threads REQUIRED)

set(SS_SFM_SRC ${SS_SRC}/sfm)
set(SS_SFM_SHADERS ${SS_SFM_SRC}/shaders)

# ---------------- 着色器变体矩阵 ----------------
# 每个缓存的 (Real, Loss) 组合生成一个 BA 模块；Windows 上 slangc 编译 ba_df_* 会中止，因此默认禁用 df。
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

# -I<shaders> 允许按相对着色器目录的路径包含共享数学代码，如 common/camera.slang，与 C++ 约定一致。
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
            # 先生成 .raw.spv 再添加修饰；slangc 不输出 NoContraction，部分驱动会合并浮点表达式，破坏双浮点模拟所需的无误差变换。
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

# 单模块阶段依次指定模块名、源文件、依赖扫描目录和逗号分隔的宏。
# 匹配器为 128/256 字节描述子分别编译启用、禁用整数点积的版本，共四种。
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

# 通过清单传递嵌入输入，避免命令行过长。
set(_sfm_listfile ${_sfm_spirv_dir}/blobs.txt)
string(REPLACE ";" "\n" _sfm_listbody "${_sfm_blobs}")
ss_write_if_different(${_sfm_listfile} "${_sfm_listbody}\n")

add_custom_command(OUTPUT ${_sfm_embed_cpp}
    COMMAND ${SS_SFM_SPIRV_TOOL} embed --sfm ${_sfm_embed_cpp}
            --list ${_sfm_listfile}
    DEPENDS ${SS_SFM_SPIRV_TOOL} ${_sfm_blobs} ${_sfm_listfile}
    COMMENT "Embedding SfM SPIR-V (${_sfm_nblobs} blobs)"
    VERBATIM)

# ---------------- 库 ----------------
file(GLOB_RECURSE SS_SFM_SOURCES CONFIGURE_DEPENDS ${SS_SFM_SRC}/*.cpp)
list(FILTER SS_SFM_SOURCES EXCLUDE REGEX "/tests/")
list(FILTER SS_SFM_SOURCES EXCLUDE REGEX "/vk/spirv_tool\.cpp$")

add_library(ss_sfm STATIC
    ${SS_SFM_SOURCES}
    ${_sfm_embed_cpp}
    # 图像与压缩实现包含在独立库中。
    ${SS_SRC}/external/stb_image_impl.cpp
    ${SS_SRC}/external/stb_image_write_impl.cpp
    ${SS_SRC}/core/ExrImage.cpp
    ${SS_SRC}/core/SceneAlign.cpp
    ${SS_SRC}/external/miniz.c
)
target_include_directories(ss_sfm PUBLIC ${SS_SRC})
target_link_libraries(ss_sfm PUBLIC ss_vulkan Threads::Threads ss_i18n)

# 此独立构建禁用学习前端；仍接受其选项名，并在运行时明确报错。
target_compile_definitions(ss_sfm PUBLIC SS_HAVE_ALIKED=0 SS_HAVE_LOMA=0)
target_compile_options(ss_sfm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${SPLAT_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:C>:${SPLAT_C_FLAGS}>)
set_property(TARGET ss_sfm PROPERTY CXX_STANDARD 17)

# 流水线大多仅含头文件；显式列出头文件供 IDE 与 Ninja 识别。
file(GLOB_RECURSE SS_SFM_HEADERS CONFIGURE_DEPENDS ${SS_SFM_SRC}/*.h)
target_sources(ss_sfm PRIVATE ${SS_SFM_HEADERS})

# ---------------- 测试：每个文件一个可执行程序 ----------------
file(GLOB SS_SFM_TESTS CONFIGURE_DEPENDS ${SS_SFM_SRC}/tests/*.cpp)
foreach(test_src ${SS_SFM_TESTS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE ss_sfm)
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 17)
    target_compile_options(${test_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${SPLAT_CXX_FLAGS}>)
endforeach()
