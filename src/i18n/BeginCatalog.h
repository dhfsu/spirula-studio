// 消息目录的语言标签宏，须与 EndCatalog.h 配对使用；不使用 #pragma once，使同一翻译单元可包含多个目录且标签不会泄漏。
// 宏无法定义宏，因此标签手工列出；末尾静态检查保证 SS_LANGUAGES 新增语言时必须同步补充标签。

#include "i18n/Message.h"

#define SS_I18N_TR(lang, s) ::spirula::i18n::Tr<::spirula::i18n::Lang::lang>{s}

#define EN(s)      SS_I18N_TR(en, s)
#define JA(s)      SS_I18N_TR(ja, s)
#define ZH_HANS(s) SS_I18N_TR(zh_hans, s)
#define ZH_HANT(s) SS_I18N_TR(zh_hant, s)
#define KO(s)      SS_I18N_TR(ko, s)
#define DE(s)      SS_I18N_TR(de, s)
#define FR(s)      SS_I18N_TR(fr, s)
#define ES(s)      SS_I18N_TR(es, s)
#define PT(s)      SS_I18N_TR(pt, s)
#define IT(s)      SS_I18N_TR(it, s)
#define NL(s)      SS_I18N_TR(nl, s)
#define RU(s)      SS_I18N_TR(ru, s)
#define TR(s)      SS_I18N_TR(tr, s)

#ifndef SS_I18N_TAG_CANARY
#define SS_I18N_TAG_CANARY
namespace spirula {
namespace i18n {
namespace detail {
inline constexpr Msg kTagCanary{
    "kTagCanary",
    EN("x"), JA("x"), ZH_HANS("x"), ZH_HANT("x"), KO("x"), DE("x"), FR("x"),
    ES("x"), PT("x"), IT("x"), NL("x"), RU("x"), TR("x")};
static_assert(kTagCanary.complete(),
              "i18n: src/i18n/BeginCatalog.h is missing a tag macro for a "
              "language in SS_LANGUAGES -- add it there too");
}  // 命名空间 detail
}  // 命名空间 i18n
}  // 命名空间 spirula
#endif
