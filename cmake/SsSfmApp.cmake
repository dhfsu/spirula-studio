# The minimal application target for SS_SFM_ONLY.

set(_ss_sfm_app_sources
    ${SS_SRC}/app/Main.cpp
    ${SS_SRC}/app/AppPaths.cpp
    ${SS_SRC}/app/CrashLog.cpp
    ${SS_SRC}/app/cli/sfm_main.cpp
    ${SS_SRC}/app/cli/sfm_ba.cpp)

if(WIN32)
    list(APPEND _ss_sfm_app_sources ${SS_SRC}/app/utf8.manifest)
endif()

add_executable(spirula-sfm ${_ss_sfm_app_sources})
target_include_directories(spirula-sfm PRIVATE ${SS_SRC} ${CMAKE_BINARY_DIR})
target_link_libraries(spirula-sfm PRIVATE ss_sfm ss_i18n)
target_compile_definitions(spirula-sfm PRIVATE
    SS_TOOL_SFM=1
    SS_VERSION="${SS_VERSION}"
    SS_DEFAULT_LANG=${SS_DEFAULT_LANG})
if(SS_FONT_CJK STREQUAL "none")
    target_compile_definitions(spirula-sfm PRIVATE SS_FONT_CJK_NONE=1)
endif()
if(WIN32)
    target_link_libraries(spirula-sfm PRIVATE dbghelp)
endif()
target_compile_options(spirula-sfm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${SPLAT_CXX_FLAGS}>)
set_property(TARGET spirula-sfm PROPERTY CXX_STANDARD 17)

if(APPLE)
    set_property(TARGET spirula-sfm PROPERTY BUILD_RPATH "@loader_path")
else()
    set_property(TARGET spirula-sfm PROPERTY BUILD_RPATH "$ORIGIN")
endif()
