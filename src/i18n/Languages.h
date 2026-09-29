#pragma once

// 语言集合的唯一来源，Lang、Msg 槽位、语言菜单及解析均据此定义；缺少译文时编译失败。
// X(id, code, native, english)：id 为 C++ 标识符及 SS_DEFAULT_LANG 值，code 用于命令行和区域设置，native 为语言自称，english 用于英文帮助及诊断。
// id 使用下划线，parse_lang 同时接受连字符；菜单顺序为英语、CJK、拉丁字母语言、西里尔字母语言。

#define SS_LANGUAGES(X)                                                       \
    X(en,      "en",      "English",     "English")                           \
    X(ja,      "ja",      "日本語",       "Japanese")                          \
    X(zh_hans, "zh-Hans", "简体中文",     "Chinese (Simplified)")               \
    X(zh_hant, "zh-Hant", "繁體中文",     "Chinese (Traditional)")              \
    X(ko,      "ko",      "한국어",       "Korean")                            \
    X(de,      "de",      "Deutsch",     "German")                            \
    X(fr,      "fr",      "Français",    "French")                            \
    X(es,      "es",      "Español",     "Spanish")                           \
    X(pt,      "pt",      "Português",   "Portuguese")                        \
    X(it,      "it",      "Italiano",    "Italian")                           \
    X(nl,      "nl",      "Nederlands",  "Dutch")                             \
    X(ru,      "ru",      "Русский",     "Russian")                           \
    X(tr,      "tr",      "Türkçe",      "Turkish")

// 语言菜单用跨文字图标“文A”加当前语言自称，便于看不懂当前界面的用户找到切换入口；嵌入字体须包含这两个字符。
#define SS_LANG_MENU_ICON "文A"

// 需要 CJK 字体的语言，用于字体加载与默认语言、字体配置的一致性检查。
#define SS_LANGUAGES_CJK(X)                                                   \
    X(ja) X(zh_hans) X(zh_hant) X(ko)
