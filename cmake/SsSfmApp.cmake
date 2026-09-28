set(_ss_sfm_app_sources
    ${SS_SRC}/app/Main.cpp
    ${SS_SRC}/app/AppPaths.cpp
    ${SS_SRC}/app/CrashLog.cpp
    ${SS_SRC}/app/cli/sfm_main.cpp
    ${SS_SRC}/app/cli/sfm_ba.cpp)

add_executable(spirula-sfm ${_ss_sfm_app_sources})
target_include_directories(spirula-sfm PRIVATE ${SS_SRC} ${CMAKE_BINARY_DIR})
target_link_libraries(spirula-sfm PRIVATE ss_sfm ss_i18n)
target_compile_definitions(spirula-sfm PRIVATE
    SS_TOOL_SFM=1
    SS_VERSION="${SS_VERSION}"
    SS_DEFAULT_LANG=${SS_DEFAULT_LANG})
target_compile_options(spirula-sfm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${SPLAT_CXX_FLAGS}>)
set_property(TARGET spirula-sfm PROPERTY CXX_STANDARD 17)
