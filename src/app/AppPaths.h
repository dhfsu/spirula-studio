#pragma once

// 统一解析应用设置目录、缓存目录及当前可执行文件的位置。
// GUI 通过自身路径启动重建子进程；设置与大型模型缓存分开存放。

#include <string>

namespace app {

// 首次调用时创建；Windows 使用漫游配置目录，Linux 使用 XDG_CONFIG_HOME。
std::string config_dir();

// 首次调用时创建；Windows 使用 LOCALAPPDATA，Linux 使用 XDG_CACHE_HOME；大型下载独立于配置目录。
std::string cache_dir();

// 当前可执行文件及其目录仅解析一次，无法确定时均为空；启动工具必须使用 exe_path()，防止 PATH 指向其他构建。
std::string exe_path();
std::string exe_dir();

// 仅用于 macOS；Finder 继承 launchd 的精简 PATH，无法找到常见第三方工具，因此在末尾追加搜索路径，保留已有路径优先级。
void add_desktop_search_paths();

}  // 命名空间 app
