#pragma once
// 共享的 Vulkan 设备选择与物理设备标识，各运行时在自己的 VkInstance 中解析。
// 优先显式选择，其次 SS_VK_DEVICE，最后 Auto；传递 UUID 以抵抗枚举顺序变化。
// 仅含头文件，仅 Vulkan 原生目标包含，避免将 Vulkan 依赖引入 CUDA 公共源文件。

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <climits>
#include "core/Env.h"

namespace spirula {
namespace vkselect {

// index 仅在当前运行时枚举中有效，跨运行时须传递 uuid；usable 表示本运行时的基本要求，不保证支持特定模型。
struct DeviceRecord {
    int                        index = -1;
    std::string                name;
    std::string                type;  // 设备类型标识：discrete|integrated|virtual|cpu|other
    uint64_t                   vram_bytes = 0;
    bool                       usable = false;
    std::string                unusable_reason;
    VkPhysicalDeviceProperties props{};
    uint8_t                    uuid[VK_UUID_SIZE] = {};
};

inline const char* deviceTypeName(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
        default:                                     return "other";
    }
}

// Auto 首先按类型排序：独立 GPU > 集成 GPU > 虚拟 GPU > CPU > 其他。
inline int deviceTypeRank(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 4;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 1;
        default:                                     return 0;
    }
}

inline bool uuidIsZero(const uint8_t uuid[VK_UUID_SIZE]) {
    for (int i = 0; i < VK_UUID_SIZE; i++)
        if (uuid[i]) return false;
    return true;
}

inline bool uuidEquals(const uint8_t a[VK_UUID_SIZE], const uint8_t b[VK_UUID_SIZE]) {
    for (int i = 0; i < VK_UUID_SIZE; i++)
        if (a[i] != b[i]) return false;
    return true;
}

// 规范选择器形式：uuid: 后接 32 个小写十六进制数字。
inline std::string uuidSelector(const uint8_t uuid[VK_UUID_SIZE]) {
    static const char* hex = "0123456789abcdef";
    std::string s = "uuid:";
    for (int i = 0; i < VK_UUID_SIZE; i++) {
        s += hex[uuid[i] >> 4];
        s += hex[uuid[i] & 0xf];
    }
    return s;
}

// 设备未报告 UUID 时为空，无法传递稳定标识。
inline std::string selectorFor(const DeviceRecord& r) {
    return uuidIsZero(r.uuid) ? std::string() : uuidSelector(r.uuid);
}

// 从枚举设备填入 name/type/props/uuid；显存大小与可用性由调用方补充。
inline void probeIdentity(VkPhysicalDevice pd, DeviceRecord* r) {
    VkPhysicalDeviceIDProperties id{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &id;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    r->props = p2.properties;
    r->name = p2.properties.deviceName;
    r->type = deviceTypeName(p2.properties.deviceType);
    std::copy(id.deviceUUID, id.deviceUUID + VK_UUID_SIZE, r->uuid);
}

// text 保留原始请求，error 说明格式错误，使 CLI 可在启动 GPU 前报告无效选项。
struct Request {
    enum class Kind { Auto, Ordinal, Name, Uuid, Malformed };
    Kind        kind = Kind::Auto;
    int         ordinal = -1;
    std::string text;
    std::string error;
    bool        explicit_request = false;  // 来自调用方而非环境变量
};

// 必须完整解析整个值；溢出、除 SfM 的 -1 外的负数及其他非法形式均视为格式错误。
inline Request parseRequest(const std::string& value) {
    Request r;
    r.text = value;
    size_t b = 0, e = value.size();
    while (b < e && std::isspace((unsigned char)value[b])) b++;
    while (e > b && std::isspace((unsigned char)value[e - 1])) e--;
    const std::string v = value.substr(b, e - b);
    r.text = v;

    if (v.empty()) {
        r.kind = Request::Kind::Malformed;
        r.error = "empty device selector";
        return r;
    }
    std::string lower = v;
    for (auto& c : lower) c = (char)std::tolower((unsigned char)c);

    if (lower == "auto" || v == "-1") {
        r.kind = Request::Kind::Auto;
        return r;
    }
    if (lower.rfind("uuid:", 0) == 0) {
        const std::string hex = v.substr(5);
        if (hex.size() != 2 * VK_UUID_SIZE) {
            r.kind = Request::Kind::Malformed;
            r.error = "uuid selector needs " +
                      std::to_string(2 * VK_UUID_SIZE) + " hex digits, got " +
                      std::to_string(hex.size());
            return r;
        }
        for (char c : hex)
            if (!std::isxdigit((unsigned char)c)) {
                r.kind = Request::Kind::Malformed;
                r.error = "uuid selector is not hex";
                return r;
            }
        r.kind = Request::Kind::Uuid;
        return r;
    }
    if (std::isdigit((unsigned char)v[0])) {
        long long n = 0;
        for (char c : v) {
            if (!std::isdigit((unsigned char)c)) {
                r.kind = Request::Kind::Malformed;
                r.error = "device index '" + v + "' is not a whole number";
                return r;
            }
            const int digit = c - '0';
            if (n > (INT32_MAX - digit) / 10) {
                r.kind = Request::Kind::Malformed;
                r.error = "device index '" + v + "' is out of range";
                return r;
            }
            n = n * 10 + digit;
        }
        r.kind = Request::Kind::Ordinal;
        r.ordinal = (int)n;
        return r;
    }
    if (v[0] == '-') {
        r.kind = Request::Kind::Malformed;
        r.error = "negative device value '" + v + "' (auto is -1 or auto)";
        return r;
    }
    r.kind = Request::Kind::Name;
    return r;
}

// 显式 CLI/GUI 值优先，其次非空 SS_VK_DEVICE，再其次 Auto；explicit_set 区分显式 Auto，避免空选择器误退回环境设置。
inline Request requestFrom(const std::string& explicit_selector, bool explicit_set) {
    if (explicit_set) {
        Request r = parseRequest(explicit_selector.empty() ? "auto"
                                                           : explicit_selector);
        r.explicit_request = true;
        return r;
    }
    const char* env = spirula::env("VK_DEVICE");
    if (env && env[0]) return parseRequest(env);
    return parseRequest("auto");
}

enum class ResolveStatus {
    Ok,
    Malformed,
    OutOfRange,
    Missing,
    Unusable,
    NoDevice,
};

struct Resolution {
    ResolveStatus status = ResolveStatus::Ok;
    std::string   error;     // 面向用户的原因，成功时为空
    DeviceRecord  device;    // 解析到的记录；不可用时仍保留匹配项
    std::string   selector;  // 规范 uuid:<hex>，未报告时为空
    bool          ok() const { return status == ResolveStatus::Ok; }
};

// 返回首个 UUID 匹配项，否则为 -1；重复安装 ICD 清单会导致重复枚举，各运行时统一采用首项。
inline int findByUuid(const std::vector<DeviceRecord>& devices,
                      const uint8_t uuid[VK_UUID_SIZE]) {
    for (size_t i = 0; i < devices.size(); i++)
        if (!uuidIsZero(devices[i].uuid) && uuidEquals(devices[i].uuid, uuid))
            return (int)i;
    return -1;
}

// 设备解析会在多个运行时和工作线程中重复，消息仅输出一次。
inline void warnOnce(const std::string& what) {
    static std::mutex m;
    static std::set<std::string> said;
    std::lock_guard<std::mutex> lock(m);
    if (said.insert(what).second) std::fprintf(stderr, "warning: %s\n", what.c_str());
}

// Auto 按设备类型、显存依次排序；未报告 deviceUUID 的设备不参与选择，Vulkan 允许全零 UUID，但其不能提供有效标识。
inline bool autoCandidate(const DeviceRecord& r) {
    return r.usable && !uuidIsZero(r.uuid);
}

inline bool outranks(const DeviceRecord& a, const DeviceRecord& b) {
    const int ra = deviceTypeRank(a.props.deviceType);
    const int rb = deviceTypeRank(b.props.deviceType);
    if (ra != rb) return ra > rb;
    return a.vram_bytes > b.vram_bytes;
}

// 返回带有效标识的最佳记录，否则为 -1。
inline int autoPick(const std::vector<DeviceRecord>& devices) {
    int best = -1;
    for (size_t i = 0; i < devices.size(); i++) {
        if (!autoCandidate(devices[i])) continue;
        if (best < 0 || outranks(devices[i], devices[best])) best = (int)i;
    }
    return best;
}

inline Resolution resolveRequest(const Request& r,
                                 const std::vector<DeviceRecord>& devices) {
    Resolution out;
    auto fail = [&](ResolveStatus s, const std::string& why) {
        out.status = s;
        out.error = why;
        return out;
    };

    if (r.kind == Request::Kind::Malformed)
        return fail(ResolveStatus::Malformed, r.error);

    int picked = -1;
    if (r.kind == Request::Kind::Auto) {
        picked = autoPick(devices);
        if (picked < 0) {
            for (const DeviceRecord& d : devices)
                if (d.usable && uuidIsZero(d.uuid))
                    return fail(ResolveStatus::Missing,
                                "device '" + d.name +
                                    "' reports no physical-device UUID, so it "
                                    "cannot be selected by identity");
            return fail(ResolveStatus::NoDevice,
                        "no usable Vulkan device (need Vulkan 1.2 + "
                        "bufferDeviceAddress + timelineSemaphore)");
        }
    } else if (r.kind == Request::Kind::Ordinal) {
        if (r.ordinal >= (int)devices.size())
            return fail(ResolveStatus::OutOfRange,
                        "device index " + std::to_string(r.ordinal) +
                        " does not exist (" + std::to_string(devices.size()) +
                        " device(s))");
        picked = r.ordinal;
    } else if (r.kind == Request::Kind::Uuid) {
        uint8_t want[VK_UUID_SIZE] = {};
        const std::string hex = r.text.substr(5);
        for (int i = 0; i < VK_UUID_SIZE; i++) {
            auto nibble = [](char c) {
                return (uint8_t)(c <= '9' ? c - '0'
                                          : std::tolower((unsigned char)c) - 'a' + 10);
            };
            want[i] = (uint8_t)(nibble(hex[2 * i]) << 4 | nibble(hex[2 * i + 1]));
        }
        picked = findByUuid(devices, want);
        if (picked < 0)
            return fail(ResolveStatus::Missing,
                        "no Vulkan device has " + r.text);
    } else {
        int found = 0;
        for (size_t i = 0; i < devices.size(); i++) {
            std::string name = devices[i].name, needle = r.text;
            for (auto& c : name) c = (char)std::tolower((unsigned char)c);
            for (auto& c : needle) c = (char)std::tolower((unsigned char)c);
            if (name.find(needle) == std::string::npos) continue;
            if (found == 0) picked = (int)i;
            found++;
        }
        if (found == 0)
            return fail(ResolveStatus::Missing,
                        "no Vulkan device name contains '" + r.text + "'");
        if (found > 1)
            warnOnce("device name '" + r.text + "' matches " +
                     std::to_string(found) + " devices; using the first, device " +
                     std::to_string(picked) + " (" + devices[picked].name + ")");
    }

    // 序号或名称仅用于查找，调用方保存的必须是稳定标识；缺少标识的记录不能满足请求。
    if (uuidIsZero(devices[picked].uuid))
        return fail(ResolveStatus::Missing,
                    "device " + std::to_string(picked) + " (" +
                        devices[picked].name +
                        ") reports no physical-device UUID, so it cannot be "
                        "selected by identity");
    int copies = 0;
    for (const DeviceRecord& o : devices)
        if (uuidEquals(o.uuid, devices[picked].uuid)) copies++;
    if (copies > 1) {
        picked = findByUuid(devices, devices[picked].uuid);
        warnOnce(std::to_string(copies) + " Vulkan devices report " +
                 uuidSelector(devices[picked].uuid) + "; using the first, device " +
                 std::to_string(picked) + " (" + devices[picked].name + ")");
    }
    const DeviceRecord& d = devices[picked];
    if (!d.usable) {
        std::string why = "device " + std::to_string(picked) + " (" + d.name +
                          ") is not usable";
        if (!d.unusable_reason.empty()) why += ": " + d.unusable_reason;
        out.device = d;
        out.status = ResolveStatus::Unusable;
        out.error = why;
        return out;
    }
    out.device = d;
    out.selector = uuidSelector(d.uuid);
    return out;
}

inline Resolution resolveSelector(const std::string& value,
                                  const std::vector<DeviceRecord>& devices) {
    return resolveRequest(parseRequest(value), devices);
}

}  // 命名空间 vkselect
}  // 命名空间 spirula
