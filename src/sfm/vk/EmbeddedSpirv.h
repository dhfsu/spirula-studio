// 查询构建时嵌入的 SPIR-V，由 SsSfm.cmake 生成翻译单元并编入 ss_sfm。
// 模块名为 .spv 主干，如 ba_double_huber，可用变体由 SS_SFM_REALS/SS_SFM_LOSSES 决定。
#pragma once

#include <cstddef>
#include <cstdint>

namespace sfm {

// 未编入的变体返回 nullptr，可选 words 接收以 32 位字为单位的长度。
const uint32_t* findSpirv(const char* name, size_t* words);

// 枚举模块名，超出范围返回 nullptr。
const char* spirvBlobName(size_t i);

}  // 命名空间 sfm
