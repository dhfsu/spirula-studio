# 为各自持有上下文的模块提供统一 Vulkan 链接目标 ss_vulkan。
# macOS 默认静态链接 MoltenVK，使程序仅依赖 /System 和 /usr/lib，避免依赖构建机上的 Homebrew 绝对路径、驱动清单和加载器。
# 静态构建无法加载验证层，需用 -DSS_MACOS_VULKAN=loader 启用；其他平台使用 FindVulkan 找到的加载器。

set(SS_MOLTENVK_VERSION "1.4.2")
set(SS_MOLTENVK_DIR "" CACHE PATH
    "Unpacked MoltenVK release (holds MoltenVK/include and MoltenVK/static); \
empty = fetch the pinned release")

# _ss_fetch_moltenvk(<lib_var> <include_var>) 查找固定版本的 MoltenVK，缺失时下载到构建目录。
# 上游包提供通用架构静态库及 Vulkan 头文件，避免依赖仅含 arm64 的 Homebrew 库或本机 Vulkan SDK。
function(_ss_fetch_moltenvk lib_var include_var)
    set(_root "${SS_MOLTENVK_DIR}")
    if(NOT _root)
        set(_name "MoltenVK-${SS_MOLTENVK_VERSION}-macos")
        set(_root ${CMAKE_BINARY_DIR}/${_name})
        if(NOT EXISTS ${_root}/MoltenVK/static)
            set(_url "https://github.com/KhronosGroup/MoltenVK/releases/download/v${SS_MOLTENVK_VERSION}/MoltenVK-macos.tar")
            set(_tar ${CMAKE_BINARY_DIR}/${_name}.tar)
            message(STATUS "Downloading ${_url} (~60 MB, once)")
            file(DOWNLOAD ${_url} ${_tar} SHOW_PROGRESS STATUS _dl)
            list(GET _dl 0 _dl_code)
            if(NOT _dl_code EQUAL 0)
                file(REMOVE ${_tar})
                message(FATAL_ERROR "Failed to download MoltenVK "
                    "${SS_MOLTENVK_VERSION}: ${_dl}. Unpack the release by "
                    "hand and pass -DSS_MOLTENVK_DIR=/path/to/MoltenVK, or "
                    "build against the loader with -DSS_MACOS_VULKAN=loader.")
            endif()
            # 发行包内容位于 MoltenVK/；仅保留构建所需的两部分，删除约 25 MB 的动态框架与 iOS 切片。
            file(ARCHIVE_EXTRACT INPUT ${_tar} DESTINATION ${_root}
                PATTERNS "MoltenVK/MoltenVK/include/*"
                         "MoltenVK/MoltenVK/static/*")
            file(REMOVE ${_tar})
            file(RENAME ${_root}/MoltenVK/MoltenVK ${_root}/.mvk_tmp)
            file(REMOVE_RECURSE ${_root}/MoltenVK)
            file(RENAME ${_root}/.mvk_tmp ${_root}/MoltenVK)
        endif()
    endif()

    set(_lib ${_root}/MoltenVK/static/MoltenVK.xcframework/macos-arm64_x86_64/libMoltenVK.a)
    set(_inc ${_root}/MoltenVK/include)
    if(NOT EXISTS ${_lib} OR NOT EXISTS ${_inc}/vulkan/vulkan.h)
        message(FATAL_ERROR "MoltenVK package at ${_root} is missing "
            "MoltenVK/static/.../libMoltenVK.a or MoltenVK/include/vulkan. "
            "Point -DSS_MOLTENVK_DIR at an unpacked MoltenVK-macos release.")
    endif()
    set(${lib_var} ${_lib} PARENT_SCOPE)
    set(${include_var} ${_inc} PARENT_SCOPE)
endfunction()

# SfM 设备能力查询要求足够新的 Vulkan 头文件。
set(SS_VULKAN_HEADERS_MIN 277)
set(SS_VULKAN_HEADERS_VERSION "1.4.357")

function(_ss_vulkan_header_version inc out_var)
    set(${out_var} 0 PARENT_SCOPE)
    if(NOT EXISTS ${inc}/vulkan/vulkan_core.h)
        return()
    endif()
    file(STRINGS ${inc}/vulkan/vulkan_core.h _line
        REGEX "^#define VK_HEADER_VERSION ")
    if(_line MATCHES "VK_HEADER_VERSION[ \t]+([0-9]+)")
        set(${out_var} ${CMAKE_MATCH_1} PARENT_SCOPE)
    endif()
endfunction()

# 将固定版本 Vulkan-Headers 解压到构建目录；纯头文件且与架构无关，适用于加载器较新但系统头文件过旧的平台。
function(_ss_fetch_vulkan_headers out_var)
    set(_name "Vulkan-Headers-${SS_VULKAN_HEADERS_VERSION}")
    set(_inc ${CMAKE_BINARY_DIR}/${_name}/include)
    if(NOT EXISTS ${_inc}/vulkan/vulkan_core.h)
        set(_url "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v${SS_VULKAN_HEADERS_VERSION}.tar.gz")
        set(_tar ${CMAKE_BINARY_DIR}/${_name}.tar.gz)
        message(STATUS "Downloading ${_url}")
        file(DOWNLOAD ${_url} ${_tar} STATUS _dl)
        list(GET _dl 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            file(REMOVE ${_tar})
            message(FATAL_ERROR "Failed to download Vulkan-Headers "
                "${SS_VULKAN_HEADERS_VERSION}: ${_dl}. Install a Vulkan SDK "
                "with headers >= ${SS_VULKAN_HEADERS_MIN}.")
        endif()
        file(ARCHIVE_EXTRACT INPUT ${_tar} DESTINATION ${CMAKE_BINARY_DIR})
        file(REMOVE ${_tar})
    endif()
    set(${out_var} ${_inc} PARENT_SCOPE)
endfunction()

# ss_vulkan_lib() 幂等定义 ss_vulkan 接口目标；多个模块均可调用，以首次调用为准。
function(ss_vulkan_lib)
    if(TARGET ss_vulkan)
        return()
    endif()
    add_library(ss_vulkan INTERFACE)

    if(APPLE AND SS_MACOS_VULKAN STREQUAL "static")
        _ss_fetch_moltenvk(_mvk_lib _mvk_inc)
        message(STATUS "Vulkan: MoltenVK ${SS_MOLTENVK_VERSION} linked "
            "statically (${_mvk_lib})")
        target_link_libraries(ss_vulkan INTERFACE ${_mvk_lib}
            "-framework Metal"       # what libMoltenVK.a leaves undefined
            "-framework Foundation"
            "-framework QuartzCore"
            "-framework IOSurface"
            "-framework IOKit"
            "-framework CoreGraphics"
            "-framework AppKit")
        target_include_directories(ss_vulkan SYSTEM INTERFACE ${_mvk_inc})
        # 无加载器时缺少 VK_KHR_portability_enumeration，VulkanContext 会在扩展不存在时跳过它。
        return()
    endif()

    find_package(Vulkan REQUIRED)
    target_link_libraries(ss_vulkan INTERFACE Vulkan::Vulkan)

    _ss_vulkan_header_version("${Vulkan_INCLUDE_DIR}" _vk_hdr)
    if(_vk_hdr LESS SS_VULKAN_HEADERS_MIN)
        message(STATUS "Vulkan headers at ${Vulkan_INCLUDE_DIR} are version "
            "${_vk_hdr} (need ${SS_VULKAN_HEADERS_MIN}); fetching "
            "Vulkan-Headers ${SS_VULKAN_HEADERS_VERSION}")
        _ss_fetch_vulkan_headers(_vk_inc)
        target_include_directories(ss_vulkan SYSTEM BEFORE INTERFACE ${_vk_inc})
    endif()
endfunction()
