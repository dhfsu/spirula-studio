#pragma once

// 各工具崩溃时在 <config>/crash.log 留下调用栈，Windows 可用消息框提示文件位置。
// 窗口程序没有终端，子进程故障通常只向父进程返回状态码，因此需要独立报告。
// 栈帧使用 module+0xRVA 表示，可由 addr2line 或匹配的 PDB 解析；含调试符号时也输出名称和文件行号。

#include <string>

namespace app {

// 尽早安装处理器，crash.log 写入 dir；安装前的崩溃无法记录。SS_CRASH_TEST=segv|throw|worker 可主动触发故障验证处理器。
void install_crash_log(const std::string& dir);

// 仅窗口程序在 Windows 上弹框显示报告路径；终端工具或 GUI 子进程可直接打印，避免模态框阻塞后台任务。
void set_crash_dialog(bool on);

// <dir>/crash.log；install_crash_log() 调用前为空。
std::string crash_log_path();

// 报告附带当前操作，便于在无符号调用栈中确定出错的数据集；并发更新最多使该行混乱，不会导致崩溃。
void set_crash_note(const std::string& what);

}  // 命名空间 app
