#pragma once
// 查询物理内存，平台不支持时返回 0，调用方自行提供回退值。

#include <cstddef>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN   // rpcndr.h 的 small 宏可能污染此处的标识符
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace sfm {

inline size_t physicalRamBytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return (size_t)st.ullTotalPhys;
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) return (size_t)pages * (size_t)page;
#endif
    return 0;
}

}  // 命名空间 sfm
