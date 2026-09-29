#pragma once

// 界面语言按优先级选择：命令行 --lang、SS_LANG、保存的 GUI 设置、系统区域、编译期 SS_DEFAULT_LANG。
// 最后一项是构建指定的语言，不固定为英语；例如日语默认构建在无 LANG 的容器中仍应显示日语。

#include "i18n/Message.h"

// 编译期默认语言使用标识符而非字符串，使拼写错误直接导致明确的编译错误。
#ifndef SS_DEFAULT_LANG
#define SS_DEFAULT_LANG en
#endif

namespace spirula {
namespace i18n {

inline constexpr Lang kDefaultLang = Lang::SS_DEFAULT_LANG;

// 规范语言码，如 zh-Hans，用于 --lang 输出与设置文件。
const char* code(Lang l);
// 语言自称，如“简体中文”，供选择器使用，不再翻译。
const char* native_name(Lang l);
// 语言英文名称，如 Chinese (Simplified)，供帮助与诊断使用。
const char* english_name(Lang l);

// 接受 ja、ja_JP.UTF-8、zh-Hans-CN、zh_TW、pt-BR、zh_hant 等形式；无匹配时返回 false 且不改 out。
// C 与 POSIX 表示未指定偏好，不参与匹配。
bool parse_lang(const char* s, Lang* out);

// 返回系统语言；无法识别时返回 kDefaultLang。
Lang detect_os_lang();

// 按优先级解析并调用 set_current()；cli 与 saved 可为空，返回最终语言。
Lang init(const char* cli, const char* saved);

// 判断是否无法仅靠拉丁字体渲染。
bool needs_cjk_font(Lang l);

// 识别并移除 argv 中的 --lang <code> 或 --lang=<code>，返回值或 nullptr；非法值由 init() 报告并列出语言码。
const char* take_lang_arg(int* argc, char** argv);

// 保存 take_lang_arg() 的结果，供 GUI 加载设置后重新解析，保证命令行优先于保存的偏好。
const char* lang_arg();

// 帮助中每种语言占一行。
std::string language_list();

}  // 命名空间 i18n
}  // 命名空间 spirula
