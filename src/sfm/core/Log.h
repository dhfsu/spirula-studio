#pragma once

// 重建日志统一输出本地化 [tag] message，按终端显示列宽补齐标签，东亚宽字符占两列。
// 开发诊断、性能分析和自测保留英文原始输出；帮助由独立帮助模块处理。

#include "i18n/Message.h"

#include <functional>
#include <initializer_list>
#include <string>

#if defined(__GNUC__) || defined(__clang__)
#define SFM_LOG_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#define SFM_LOG_PRINTF(a, b)
#endif

namespace sfm {
namespace slog {

// 日志按用户可识别的阶段分类，而非按源文件分类。
enum class Tag {
    Run,       // auto 的运行请求与摘要
    Extract,   // 特征检测及检测器统计
    Match,     // 匹配、验证与相机分组
    Map,       // 建图及模型装配
    Merge,     // 合并分散重建模型
    Orient,    // 最终坐标规范对齐
    Device,    // GPU 选择与能力
};

// 决定输出流与前端样式；Diag 为带自身标签的英文诊断，原样输出且不补齐本地化标签列。
enum class Level { Info, Note, Warning, Error, Diag };

// 进程级日志接收端，文本不含阶段或警告前缀，Diag 除外；可调用 prefix，但不得再次记日志，否则逐行锁会死锁。
using Sink = std::function<void(Tag, Level, const std::string&)>;
void set_sink(Sink s);   // 空接收端恢复默认打印行为

// 返回本地化且等宽的方括号标签；按值返回，避免语言切换重建缓存时与其他线程读取竞争。
std::string prefix(Tag t);

// UTF-8 显示列宽，东亚宽字符占两列；实现复用 i18n::display_width，供摘要和表格对齐。
int display_width(const char* s);

// 输出带标签的本地化单行，args 填充位置占位符。
void out(Tag t, const spirula::i18n::Msg& m,
         std::initializer_list<spirula::i18n::Arg> args = {});
void err(Tag t, const spirula::i18n::Msg& m,
         std::initializer_list<spirula::i18n::Arg> args = {});
// 在消息前添加本地化警告或错误词，均输出到 stderr。
void warn(Tag t, const spirula::i18n::Msg& m,
          std::initializer_list<spirula::i18n::Arg> args = {});
void fail(Tag t, const spirula::i18n::Msg& m,
          std::initializer_list<spirula::i18n::Arg> args = {});

// 按显示精度格式化数字，如 num(1.14962,2) 得到 1.15，避免默认 %g 输出过多无用精度。
std::string num(double v, int decimals);

// 已格式化路径、数值或文本原样输出，保持标签列；Raw 显式表示不翻译。
void out_raw(Tag t, const std::string& text);
void err_raw(Tag t, const std::string& text);

// printf 格式开发诊断，原样写 stderr，自身不添加换行；fmt 应含英文标签，不参与本地化列对齐。
void diag(Tag t, const char* fmt, ...) SFM_LOG_PRINTF(2, 3);

}  // 命名空间 slog
}  // 命名空间 sfm
