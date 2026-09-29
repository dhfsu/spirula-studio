// 最小 Vulkan 计算上下文，封装设备、分块上传下载、共享描述符集、命名计算流水线与命令记录。
#pragma once


#include <chrono>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/Env.h"
#include "core/SourcePath.h"
#include "core/VulkanDeviceSelection.h"

#include "sfm/core/Log.h"
#include "i18n/catalog/Sfm.h"

// 共享设备记录保存规范标识，使 BA 能力缓存按 UUID 而非易变序号索引。
using VkDeviceRecord = spirula::vkselect::DeviceRecord;

// 为用户可处理的错误码提供名称，其余显示数值，避免显存不足仅表现为不明的 Vulkan -2。
inline const char* vkResultName(int r) {
    switch (r) {
        case -1: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case -2: return "VK_ERROR_OUT_OF_DEVICE_MEMORY (the GPU ran out of memory -- "
                        "a smaller --quality, a --vram-budget, or not sharing the device "
                        "with another job)";
        case -3: return "VK_ERROR_INITIALIZATION_FAILED";
        case -4: return "VK_ERROR_DEVICE_LOST";
        case -5: return "VK_ERROR_MEMORY_MAP_FAILED";
        case -8: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case -9: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case -11: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        default: return "";
    }
}

inline bool hasDeviceExtension(VkPhysicalDevice phys, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data());
    for (const auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

// MoltenVK 等可移植子集驱动需实例显式启用枚举，否则可能隐藏设备或报不兼容驱动；扩展缺失时跳过，exts 生命周期须覆盖实例创建。
inline void vkEnablePortability(VkInstanceCreateInfo& ici,
                                std::vector<const char*>& exts) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> avail(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, avail.data());
    for (const auto& e : avail) {
        if (std::strcmp(e.extensionName,
                        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) != 0)
            continue;
        exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        ici.enabledExtensionCount = (uint32_t)exts.size();
        ici.ppEnabledExtensionNames = exts.data();
        return;
    }
}

// 失败调用抛异常而非退出，设备丢失或分配失败可由 Bundle 层回退 CPU 继续重建。
struct VkError : std::runtime_error {
    VkError(int r, const std::string& what) : std::runtime_error(what), result(r) {}
    int result;
};

// 设备丢失或内存拒绝允许尝试主机；其他错误视为调用方问题。
inline bool vkErrorIsResourceFailure(int r) {
    return r == VK_ERROR_DEVICE_LOST || r == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
           r == VK_ERROR_OUT_OF_HOST_MEMORY;
}

inline std::string vkErrorText(int r, const char* file, int line) {
    char buf[512];
    snprintf(buf, sizeof buf, "Vulkan error %d %s at %s:%d", r, vkResultName(r), file, line);
    return buf;
}

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) \
    throw VkError((int)_r, vkErrorText((int)_r, SS_FILE, __LINE__)); } while (0)

// 推送常量布局必须与 ba.slang 一致。
struct Push {
    uint32_t u0 = 0, u1 = 0, u2 = 0, u3 = 0;
    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    // 新增字段追加在后，保持前 32 字节兼容；Vulkan 保证至少 128 字节，着色器可声明小于流水线布局的范围。
    uint32_t u4 = 0, u5 = 0, u6 = 0, u7 = 0;
};

struct GpuBuffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct VkContextOptions {
    bool needFloat64 = false;
    bool needFloatAtomics = false;   // VK_EXT_shader_atomic_float：f32 原子加，需 fp64 时还要求 f64 原子加
    bool needInt64Atomics = false;   // 缓冲 int64 原子操作，用于 DF 的 CAS
    bool needIntDotProduct = false;  // 打包 uint8x4 点积扩展
    // 下列设备序号仅为外部兼容输入，selector 为空时生效，-1 表示自动。
    int deviceIndex = -1;
    // 每上下文在自身实例中重新解析 UUID，保持跨枚举顺序变化的设备身份。
    std::string selector;
    bool validate = false;
    bool profile = false;            // 逐分派 GPU 时间戳，按流水线名称汇总
};

// 必须实际探测设备能力，不能按独显或集显推断；例如部分 Intel 集显有 int64 原子却无 fp64/f32 原子加。
// 缺失能力应在创建设备前明确报告，供 BA 选择算术路径。
struct VkDeviceCaps {
    bool float64 = false;           // shaderFloat64 能力
    bool float32AtomicAdd = false;  // 缓冲 f32 原子加能力
    bool float64AtomicAdd = false;  // 缓冲 f64 原子加能力
    bool int64Atomics = false;      // shaderInt64 与 shaderBufferInt64Atomics
    bool intDotProduct = false;     // 整数点积扩展是否可用
    // 下项另检查打包形式是否硬件加速；M2 模拟曾耗时每对 114 ms，手工展开为 43 ms，因此按加速属性选模块。
    bool intDotProductFast = false;
};

// 线程内保存最近设备请求失败原因，因失败发生在上下文创建前，不能存于尚不存在的对象。
inline std::string& lastVkSelectError() {
    static thread_local std::string e;
    return e;
}

// 能力探测仅携带设备请求，真正创建上下文的副本再加入所需特性。
inline VkContextOptions deviceOnlyOpt(int deviceIndex, const std::string& selector) {
    VkContextOptions o;
    o.deviceIndex = deviceIndex;
    o.selector = selector;
    return o;
}

class VkContext {
public:
    VkContext() = default;
    // 禁止复制上下文，避免所有 Vulkan 句柄被重复销毁。
    VkContext(const VkContext&) = delete;
    VkContext& operator=(const VkContext&) = delete;

    // 析构释放本上下文记录的全部缓冲与设备，防止周期 BA 累积泄漏；1363 图运行曾因每轮设备与问题缓冲未释放而耗尽显存。
    ~VkContext() {
        if (device_ == VK_NULL_HANDLE) {
            if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
            return;
        }
        vkDeviceWaitIdle(device_);
        for (auto& p : pipelines_) vkDestroyPipeline(device_, p.second, nullptr);
        if (shaderModule_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, shaderModule_, nullptr);
        if (descPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, descPool_, nullptr);
        if (pipeLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipeLayout_, nullptr);
        if (setLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
        if (queryPool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, queryPool_, nullptr);
        if (stagingPtr_) vkUnmapMemory(device_, staging_.mem);
        if (stagingDlPtr_) vkUnmapMemory(device_, stagingDl_.mem);
        for (auto& a : allocations_) {
            vkDestroyBuffer(device_, a.first, nullptr);
            vkFreeMemory(device_, a.second, nullptr);
        }
        if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
        if (cmdPool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, cmdPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroyInstance(instance_, nullptr);
    }

    // init 探测到的设备能力，初始化前为零。
    const VkDeviceCaps& caps() const { return caps_; }

    // 规范设备选择器与驱动名称，供缓存和日志使用，init 前为空。
    const std::string& selector() const { return selector_; }
    const std::string& deviceName() const { return deviceName_; }

    // 无需拥有上下文即可探测能力，供提前选择 BA 路径；无可用设备时全部为 false。
    static VkDeviceCaps probeCaps(const VkContextOptions& opt) {
        return probe(opt).caps;
    }
    static VkDeviceCaps probeCaps(int deviceIndex) {
        return probeCaps(deviceOnlyOpt(deviceIndex, std::string()));
    }

    // 无副作用枚举，供选择器和入口解析，不创建逻辑设备或分配缓冲。
    static std::vector<VkDeviceRecord> listDevices() {
        std::vector<VkDeviceRecord> out;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "vk_ba";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        std::vector<const char*> instExts;
        vkEnablePortability(ici, instExts);
        VkInstance inst = VK_NULL_HANDLE;
        if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return out;
        out = enumerateRecords(inst);
        vkDestroyInstance(inst, nullptr);
        return out;
    }

    // 将请求解析为 UUID，失败仅返回原因，不创建设备。
    static spirula::vkselect::Resolution resolveSelector(
        const spirula::vkselect::Request& req) {
        return spirula::vkselect::resolveRequest(req, listDevices());
    }

    // 按请求写法缓存解析结果，避免每次 BA 创建实例并枚举设备，假定运行中设备集合稳定。
    static const spirula::vkselect::Resolution& cachedSelector(
        const spirula::vkselect::Request& req) {
        static std::mutex m;
        static std::map<std::string, spirula::vkselect::Resolution> cache;
        const std::string key = std::to_string((int)req.kind) + "|" + req.text;
        std::lock_guard<std::mutex> g(m);
        auto it = cache.find(key);
        if (it == cache.end()) it = cache.emplace(key, resolveSelector(req)).first;
        return it->second;
    }

    // 对已有配置对象的设备请求执行相同缓存解析。
    static const spirula::vkselect::Resolution& cachedResolution(const VkContextOptions& opt) {
        spirula::vkselect::Request req;
        if (!opt.selector.empty()) {
            req = spirula::vkselect::parseRequest(opt.selector);
        } else if (opt.deviceIndex >= 0) {
            req = spirula::vkselect::parseRequest(std::to_string(opt.deviceIndex));
            req.explicit_request = true;
        } else {
            req = spirula::vkselect::requestFrom("", false);
        }
        return cachedSelector(req);
    }

    // 探测同时保留设备身份；resolved=false 表示选择错误，不等同于选中无特性的设备。
    struct Probe {
        VkDeviceCaps caps;
        VkDeviceRecord device;
        bool resolved = false;
    };
    static Probe probe(const VkContextOptions& opt) {
        Probe out;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "vk_ba_probe";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        std::vector<const char*> instExts;
        vkEnablePortability(ici, instExts);
        VkInstance inst = VK_NULL_HANDLE;
        if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return out;
        const VkPhysicalDevice phys = choosePhysical(inst, opt, &out.device);
        if (phys != VK_NULL_HANDLE) {
            out.caps = queryCaps(phys);
            out.resolved = true;
        }
        vkDestroyInstance(inst, nullptr);
        return out;
    }

    void init(const VkContextOptions& opt) {
        // ---------------- 实例 ----------------
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "vk_ba";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        std::vector<const char*> instExts;
        vkEnablePortability(ici, instExts);
        const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
        if (opt.validate) {
            ici.enabledLayerCount = 1;
            ici.ppEnabledLayerNames = layers;
        }
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance_));

        // ---------------- 物理设备 ----------------
        VkDeviceRecord record;
        phys_ = choosePhysical(instance_, opt, &record);
        if (phys_ == VK_NULL_HANDLE) {
            // 请求被拒绝是选择错误，解析器已提供具体原因，不能误报设备缺失。
            const std::string& why = lastVkSelectError();
            throw std::runtime_error(why.empty() ? "no Vulkan devices" : why);
        }
        std::copy(record.uuid, record.uuid + VK_UUID_SIZE, deviceUUID_);
        deviceName_ = record.name;
        selector_ = spirula::vkselect::uuidSelector(deviceUUID_);
        caps_ = queryCaps(phys_);
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        sfm::slog::out(
            sfm::slog::Tag::Device, spirula::i18n::msg::sfm::device_using,
            {deviceName_ + " [" + selector_ + "]"});

        // ---------------- 队列族 ----------------
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qf.data());
        queueFamily_ = ~0u;
        for (uint32_t i = 0; i < qn; i++)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily_ = i; break; }
        if (queueFamily_ == ~0u) throw std::runtime_error("no compute queue");

        // ---------------- 逻辑设备 ----------------
        // 创建前逐项检查所需能力，明确命名缺失特性，避免驱动仅返回 FEATURE_NOT_PRESENT。
        auto require = [&](bool want, bool have, const char* what) {
            if (want && !have)
                throw std::runtime_error(
                    std::string("device '") + props.deviceName +
                    "' does not support " + what +
                    ", which this stage needs (pick another device with "
                    "--device, or see `spirula sfm auto --help` for the "
                    "scalar-type flags)");
        };
        require(opt.needFloat64, caps_.float64, "fp64 shader arithmetic");
        require(opt.needFloatAtomics, caps_.float32AtomicAdd,
                "buffer float32 atomic add");
        require(opt.needFloatAtomics && opt.needFloat64, caps_.float64AtomicAdd,
                "buffer float64 atomic add");
        require(opt.needInt64Atomics, caps_.int64Atomics,
                "buffer int64 atomics");
        require(opt.needIntDotProduct, caps_.intDotProduct,
                "integer dot product");

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queueFamily_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        feat2.features.shaderFloat64 = opt.needFloat64 ? VK_TRUE : VK_FALSE;
        feat2.features.shaderInt64 = opt.needInt64Atomics ? VK_TRUE : VK_FALSE;

        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.shaderBufferInt64Atomics = opt.needInt64Atomics ? VK_TRUE : VK_FALSE;
        feat2.pNext = &f12;

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT fAtom{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        std::vector<const char*> exts;
        // 可移植设备声明此扩展时必须启用；使用字面名称，避免依赖仅在 beta 宏下定义的名称宏。
        if (hasDeviceExtension(phys_, "VK_KHR_portability_subset"))
            exts.push_back("VK_KHR_portability_subset");
        if (opt.needFloatAtomics) {
            fAtom.shaderBufferFloat32Atomics = VK_TRUE;
            fAtom.shaderBufferFloat32AtomicAdd = VK_TRUE;
            if (opt.needFloat64) {
                fAtom.shaderBufferFloat64Atomics = VK_TRUE;
                fAtom.shaderBufferFloat64AtomicAdd = VK_TRUE;
            }
            fAtom.pNext = feat2.pNext;
            feat2.pNext = &fAtom;
            exts.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        }

        // 描述子打包点积在 Vulkan 1.3 为核心，此处请求 1.2，因此以扩展启用（D21）。
        VkPhysicalDeviceShaderIntegerDotProductFeatures fDot{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        if (opt.needIntDotProduct) {
            fDot.shaderIntegerDotProduct = VK_TRUE;
            fDot.pNext = feat2.pNext;
            feat2.pNext = &fDot;
            exts.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
        }

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &feat2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = (uint32_t)exts.size();
        dci.ppEnabledExtensionNames = exts.data();
        VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &device_));
        vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

        vkGetPhysicalDeviceMemoryProperties(phys_, &memProps_);

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = queueFamily_;
        VK_CHECK(vkCreateCommandPool(device_, &cpi, nullptr, &cmdPool_));

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(device_, &fci, nullptr, &fence_));

        // 上传与下载使用独立暂存缓冲：独显 HOST_COHERENT 常为合并写内存，读取无缓存很慢。
        // 下载优先请求 HOST_CACHED，无对应堆时才回退共享类型。
        staging_ = createBufferRaw(kStagingSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkMapMemory(device_, staging_.mem, 0, kStagingSize, 0, &stagingPtr_));
        stagingDl_ = createBufferRaw(kStagingSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        VK_CHECK(vkMapMemory(device_, stagingDl_.mem, 0, kStagingSize, 0, &stagingDlPtr_));

        if (opt.profile) {
            profiling_ = true;
            timestampPeriod_ = props.limits.timestampPeriod;
            VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpi.queryCount = kMaxQueries;
            VK_CHECK(vkCreateQueryPool(device_, &qpi, nullptr, &queryPool_));
        }
    }

    GpuBuffer createBuffer(VkDeviceSize size) {
        GpuBuffer b = createBufferRaw(size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        totalAllocated_ += size;
        return b;
    }

    // 持久上下文允许提前释放单缓冲；submit 同步等待栅栏完成，因此返回后无设备任务仍引用它。
    void destroyBuffer(GpuBuffer& b) {
        if (b.buf == VK_NULL_HANDLE) return;
        vkDestroyBuffer(device_, b.buf, nullptr);
        vkFreeMemory(device_, b.mem, nullptr);
        for (size_t i = 0; i < allocations_.size(); i++)
            if (allocations_[i].first == b.buf) {
                allocations_.erase(allocations_.begin() + i);
                break;
            }
        totalAllocated_ -= b.size;
        b = GpuBuffer{};
    }

    bool initialized() const { return device_ != VK_NULL_HANDLE; }
    bool hasPipeline(const std::string& name) const { return pipelines_.count(name) != 0; }

    void upload(const GpuBuffer& dst, const void* src, VkDeviceSize size, VkDeviceSize dstOff = 0) {
        const uint8_t* p = (const uint8_t*)src;
        for (VkDeviceSize off = 0; off < size; off += kStagingSize) {
            VkDeviceSize chunk = std::min<VkDeviceSize>(kStagingSize, size - off);
            memcpy(stagingPtr_, p + off, chunk);
            VkCommandBuffer cb = begin();
            VkBufferCopy c{0, dstOff + off, chunk};
            vkCmdCopyBuffer(cb, staging_.buf, dst.buf, 1, &c);
            submit(cb);
        }
    }

    // 多个上传合并到一次提交，暂存装不下时刷新再继续，超大单项走分块。
    // 十七次独立栅栏上传在单线程约 3 ms，八线程争用时约 69 ms，曾成为小型原子 BA 主开销。
    struct UploadItem {
        const GpuBuffer* dst;
        const void* src;
        VkDeviceSize size;
    };
    void uploadMany(const UploadItem* items, size_t n) {
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VkDeviceSize used = 0;
        std::vector<VkBufferCopy> copies;
        std::vector<VkBuffer> dsts;
        auto flush = [&] {
            if (cb == VK_NULL_HANDLE) return;
            for (size_t i = 0; i < copies.size(); i++)
                vkCmdCopyBuffer(cb, staging_.buf, dsts[i], 1, &copies[i]);
            submit(cb);
            cb = VK_NULL_HANDLE;
            used = 0;
            copies.clear();
            dsts.clear();
        };
        for (size_t i = 0; i < n; i++) {
            const UploadItem& it = items[i];
            if (!it.dst || !it.size) continue;
            if (it.size > kStagingSize) {  // 无法整体容纳，使用分块上传
                flush();
                upload(*it.dst, it.src, it.size);
                continue;
            }
            if (used + it.size > kStagingSize) flush();
            if (cb == VK_NULL_HANDLE) cb = begin();
            memcpy((uint8_t*)stagingPtr_ + used, it.src, it.size);
            copies.push_back(VkBufferCopy{used, 0, it.size});
            dsts.push_back(it.dst->buf);
            // 源偏移保持 16 字节对齐，避免驱动走非对齐慢路径。
            used = (used + it.size + 15) & ~(VkDeviceSize)15;
        }
        flush();
    }

    void download(const GpuBuffer& src, void* dst, VkDeviceSize size, VkDeviceSize srcOff = 0) {
        uint8_t* p = (uint8_t*)dst;
        for (VkDeviceSize off = 0; off < size; off += kStagingSize) {
            VkDeviceSize chunk = std::min<VkDeviceSize>(kStagingSize, size - off);
            VkCommandBuffer cb = begin();
            VkBufferCopy c{srcOff + off, 0, chunk};
            vkCmdCopyBuffer(cb, src.buf, stagingDl_.buf, 1, &c);
            submit(cb);
            memcpy(p + off, stagingDlPtr_, chunk);
        }
    }

    // 将回读复制记录到现有命令缓冲，提交一次后直接读取映射暂存区，省独立 download 的额外栅栏；大小不得超过暂存容量。
    void recordDownload(VkCommandBuffer cb, const GpuBuffer& src, VkDeviceSize size,
                        VkDeviceSize srcOff = 0, VkDeviceSize dstOff = 0) {
        VkBufferCopy c{srcOff, dstOff, size};
        vkCmdCopyBuffer(cb, src.buf, stagingDl_.buf, 1, &c);
    }
    const void* stagingDownloadPtr() const { return stagingDlPtr_; }
    static constexpr VkDeviceSize stagingCapacity() { return kStagingSize; }

    // ---------------- 共用存储缓冲描述符集 ----------------
    // 首次创建布局、池与集合，后续相同绑定数只重写缓冲，供持久上下文换问题；仅可在同步提交之间执行，不能在记录期间修改。
    void createDescriptors(const std::vector<VkBuffer>& buffers) {
        uint32_t nb = (uint32_t)buffers.size();
        if (setLayout_ != VK_NULL_HANDLE) {
            if (nb != descBindingCount_)
                throw std::runtime_error("createDescriptors: binding count changed");
            writeDescriptors(buffers);
            return;
        }
        descBindingCount_ = nb;
        std::vector<VkDescriptorSetLayoutBinding> binds(nb);
        for (uint32_t i = 0; i < nb; i++)
            binds[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = nb;
        li.pBindings = binds.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &li, nullptr, &setLayout_));

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &setLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipeLayout_));

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nb};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(device_, &dpi, nullptr, &descPool_));

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = descPool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &setLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &dai, &descSet_));
        writeDescriptors(buffers);
    }

    void writeDescriptors(const std::vector<VkBuffer>& buffers) {
        uint32_t nb = (uint32_t)buffers.size();
        std::vector<VkDescriptorBufferInfo> infos(nb);
        std::vector<VkWriteDescriptorSet> writes(nb);
        for (uint32_t i = 0; i < nb; i++) {
            infos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = descSet_;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, nb, writes.data(), 0, nullptr);
    }

    void loadPipelines(const std::string& spvPath, const std::vector<std::string>& entries) {
        std::ifstream f(spvPath, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("cannot open " + spvPath);
        size_t sz = (size_t)f.tellg();
        f.seekg(0);
        std::vector<char> code(sz);
        f.read(code.data(), sz);
        loadPipelines((const uint32_t*)code.data(), sz, entries);
    }

    // 单模块包含命名入口，按需创建流水线并跳过已有项；冷缓存每入口约 90 ms，避免编译本问题不使用的相机模型。
    void loadPipelines(const uint32_t* code, size_t codeBytes,
                       const std::vector<std::string>& entries) {
        if (shaderModule_ == VK_NULL_HANDLE) {
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi.codeSize = codeBytes;
            smi.pCode = code;
            VK_CHECK(vkCreateShaderModule(device_, &smi, nullptr, &shaderModule_));
        }

        // SS_SFM_MAP_PROF 报告超过 100 ms 的入口编译，定位冷驱动缓存的主要开销。
        const bool prof = spirula::env("SFM_MAP_PROF") != nullptr;
        for (const auto& e : entries) {
            if (pipelines_.count(e)) continue;
            auto t0 = std::chrono::steady_clock::now();
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpi.stage.module = shaderModule_;
            cpi.stage.pName = e.c_str();
            cpi.layout = pipeLayout_;
            VkPipeline p;
            VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &p));
            pipelines_[e] = p;
            if (prof) {
                double dt = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
                if (dt > 0.1)
                    sfm::slog::diag(sfm::slog::Tag::Device, "[prof]   pipeline %s: %.2f s",
                                    e.c_str(), dt);
            }
        }
    }

    // ---------------- 命令记录 ----------------
    VkCommandBuffer begin() {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = cmdPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cb));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        if (descSet_ != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1,
                                    &descSet_, 0, nullptr);
        if (profiling_) {
            vkCmdResetQueryPool(cb, queryPool_, 0, kMaxQueries);
            queryNames_.clear();
        }
        return cb;
    }

    void dispatch(VkCommandBuffer cb, const std::string& name, uint32_t gx, const Push& push,
                  uint32_t gy = 1) {
        auto it = pipelines_.find(name);
        if (it == pipelines_.end()) throw std::runtime_error("unknown pipeline " + name);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, it->second);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        bool prof = profiling_ && 2 * queryNames_.size() + 2 <= kMaxQueries;
        if (prof)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_,
                                (uint32_t)(2 * queryNames_.size()));
        vkCmdDispatch(cb, gx, gy, 1);
        if (prof) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_,
                                (uint32_t)(2 * queryNames_.size() + 1));
            queryNames_.push_back(name);
        }
    }

    void barrier(VkCommandBuffer cb,
                 VkPipelineStageFlags src = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VkPipelineStageFlags dst = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, src | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             dst | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void fillZero(VkCommandBuffer cb, const GpuBuffer& b) {
        vkCmdFillBuffer(cb, b.buf, 0, VK_WHOLE_SIZE, 0);
    }

    // vkCmdFillBuffer 要求偏移和大小均为四字节倍数
    void fillZero(VkCommandBuffer cb, const GpuBuffer& b, VkDeviceSize offset, VkDeviceSize size) {
        vkCmdFillBuffer(cb, b.buf, offset, size, 0);
    }

    void copy(VkCommandBuffer cb, const GpuBuffer& src, const GpuBuffer& dst, VkDeviceSize size) {
        VkBufferCopy c{0, 0, size};
        vkCmdCopyBuffer(cb, src.buf, dst.buf, 1, &c);
    }

    void submit(VkCommandBuffer cb) {
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
        VK_CHECK(vkWaitForFences(device_, 1, &fence_, VK_TRUE, ~0ull));
        VK_CHECK(vkResetFences(device_, 1, &fence_));
        vkFreeCommandBuffers(device_, cmdPool_, 1, &cb);
        if (profiling_ && !queryNames_.empty()) {
            std::vector<uint64_t> ts(2 * queryNames_.size());
            VK_CHECK(vkGetQueryPoolResults(device_, queryPool_, 0, (uint32_t)ts.size(),
                                           ts.size() * 8, ts.data(), 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            for (size_t i = 0; i < queryNames_.size(); i++) {
                auto& s = profStats_[queryNames_[i]];
                s.first += 1;
                s.second += (ts[2 * i + 1] - ts[2 * i]) * (double)timestampPeriod_ * 1e-6;
            }
            queryNames_.clear();
        }
    }

    void printProfile(FILE* f = stderr) const {
        if (!profiling_) return;
        std::vector<std::pair<std::string, std::pair<uint64_t, double>>> v(profStats_.begin(),
                                                                           profStats_.end());
        std::sort(v.begin(), v.end(),
                  [](auto& a, auto& b) { return a.second.second > b.second.second; });
        double total = 0;
        for (auto& e : v) total += e.second.second;
        fprintf(f, "[profile] %-18s %10s %8s %10s\n", "kernel", "total_ms", "count", "avg_us");
        for (auto& e : v)
            fprintf(f, "[profile] %-18s %10.2f %8llu %10.1f\n", e.first.c_str(), e.second.second,
                    (unsigned long long)e.second.first, 1e3 * e.second.second / e.second.first);
        fprintf(f, "[profile] %-18s %10.2f\n", "TOTAL", total);
    }

    VkDevice device() const { return device_; }
    double totalAllocatedMB() const { return totalAllocated_ / (1024.0 * 1024.0); }
    double deviceLocalHeapMB() const {
        VkDeviceSize best = 0;
        for (uint32_t i = 0; i < memProps_.memoryHeapCount; i++)
            if (memProps_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                best = std::max(best, memProps_.memoryHeaps[i].size);
        return best / (1024.0 * 1024.0);
    }

private:
    static constexpr VkDeviceSize kStagingSize = 64ull << 20;

    // 枚举与 choosePhysical 共用可用性探测。
    static std::vector<VkDeviceRecord> enumerateRecords(
        VkInstance inst, std::vector<VkPhysicalDevice>* handles = nullptr) {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(inst, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        if (n) vkEnumeratePhysicalDevices(inst, &n, devs.data());
        if (handles) *handles = devs;

        std::vector<spirula::vkselect::DeviceRecord> list;
        list.reserve(n);
        for (uint32_t i = 0; i < n; i++) {
            spirula::vkselect::DeviceRecord r;
            r.index = (int)i;
            spirula::vkselect::probeIdentity(devs[i], &r);
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(devs[i], &mp);
            for (uint32_t h = 0; h < mp.memoryHeapCount; ++h)
                if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    r.vram_bytes += mp.memoryHeaps[h].size;
            // 按共享运行时基线提前拒绝不可用设备，避免等到 vkCreateDevice 失败。
            VkPhysicalDeviceVulkan12Features f12{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceFeatures2 f2{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &f12;
            vkGetPhysicalDeviceFeatures2(devs[i], &f2);
            uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, nullptr);
            std::vector<VkQueueFamilyProperties> qf(qn);
            if (qn) vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qf.data());
            bool compute = false;
            for (uint32_t q = 0; q < qn; q++)
                if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) compute = true;
            if (r.props.apiVersion < VK_API_VERSION_1_2)
                r.unusable_reason = "driver reports Vulkan < 1.2";
            else if (!f12.bufferDeviceAddress)
                r.unusable_reason = "no bufferDeviceAddress";
            else if (!f12.timelineSemaphore)
                r.unusable_reason = "no timelineSemaphore";
            else if (!compute)
                r.unusable_reason = "no compute queue";
            else
                r.usable = true;
            list.push_back(std::move(r));
        }

        return list;
    }

    static VkPhysicalDevice choosePhysical(VkInstance inst, const VkContextOptions& opt,
                                           VkDeviceRecord* record = nullptr) {
        std::vector<VkPhysicalDevice> devs;
        const std::vector<VkDeviceRecord> list = enumerateRecords(inst, &devs);
        const spirula::vkselect::Resolution res =
            spirula::vkselect::resolveRequest(deviceRequest(opt), list);
        if (!res.ok()) {
            lastVkSelectError() = res.error;
            return VK_NULL_HANDLE;
        }
        if (record) *record = res.device;
        return devs[(size_t)res.device.index];
    }

    // 设备请求优先规范身份，其次兼容序号，最后共享环境与自动选择规则。
    static spirula::vkselect::Request deviceRequest(const VkContextOptions& opt) {
        if (!opt.selector.empty()) return spirula::vkselect::parseRequest(opt.selector);
        if (opt.deviceIndex >= 0) {
            spirula::vkselect::Request r =
                spirula::vkselect::parseRequest(std::to_string(opt.deviceIndex));
            r.explicit_request = true;
            return r;
        }
        return spirula::vkselect::requestFrom("", false);
    }

    static VkDeviceCaps queryCaps(VkPhysicalDevice phys) {
        // 仅设备声明原子浮点扩展时查询对应特性结构，不能把不支持结构链入 Features2。
        uint32_t en = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &en, nullptr);
        std::vector<VkExtensionProperties> exts(en);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &en, exts.data());
        auto has_ext = [&](const char* name) {
            for (const auto& e : exts)
                if (std::strcmp(e.extensionName, name) == 0) return true;
            return false;
        };

        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f2.pNext = &f12;
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT fAtom{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        const bool atomic_float_ext = has_ext(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        if (atomic_float_ext) { fAtom.pNext = f2.pNext; f2.pNext = &fAtom; }
        VkPhysicalDeviceShaderIntegerDotProductFeatures fDot{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        const bool dot_ext = has_ext(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
        if (dot_ext) { fDot.pNext = f2.pNext; f2.pNext = &fDot; }
        vkGetPhysicalDeviceFeatures2(phys, &f2);

        VkPhysicalDeviceShaderIntegerDotProductProperties pDot{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
        if (dot_ext) {
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext = &pDot;
            vkGetPhysicalDeviceProperties2(phys, &p2);
        }

        VkDeviceCaps c;
        c.float64 = f2.features.shaderFloat64 == VK_TRUE;
        c.float32AtomicAdd = atomic_float_ext &&
                             fAtom.shaderBufferFloat32Atomics == VK_TRUE &&
                             fAtom.shaderBufferFloat32AtomicAdd == VK_TRUE;
        c.float64AtomicAdd = atomic_float_ext &&
                             fAtom.shaderBufferFloat64Atomics == VK_TRUE &&
                             fAtom.shaderBufferFloat64AtomicAdd == VK_TRUE;
        c.int64Atomics = f2.features.shaderInt64 == VK_TRUE &&
                         f12.shaderBufferInt64Atomics == VK_TRUE;
        c.intDotProduct = dot_ext && fDot.shaderIntegerDotProduct == VK_TRUE;
        c.intDotProductFast =
            c.intDotProduct &&
            pDot.integerDotProduct4x8BitPackedUnsignedAccelerated == VK_TRUE;
        return c;
    }

    // 先同时满足必需和偏好内存标志，无匹配时仅保留必需标志。
    GpuBuffer createBufferRaw(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags memFlags,
                              VkMemoryPropertyFlags preferFlags = 0) {
        GpuBuffer b;
        b.size = size;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &b.buf));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, b.buf, &req);
        uint32_t typeIdx = ~0u;
        for (int pass = preferFlags ? 0 : 1; pass < 2 && typeIdx == ~0u; pass++) {
            const VkMemoryPropertyFlags want = pass == 0 ? (memFlags | preferFlags) : memFlags;
            for (uint32_t i = 0; i < memProps_.memoryTypeCount; i++)
                if ((req.memoryTypeBits & (1u << i)) &&
                    (memProps_.memoryTypes[i].propertyFlags & want) == want) {
                    typeIdx = i;
                    break;
                }
        }
        if (typeIdx == ~0u) throw std::runtime_error("no suitable memory type");
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = typeIdx;
        VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &b.mem));
        VK_CHECK(vkBindBufferMemory(device_, b.buf, b.mem, 0));
        allocations_.emplace_back(b.buf, b.mem);
        return b;
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDeviceCaps caps_{};
    std::string selector_;   // phys_ 的规范 uuid:<hex>，init 前为空
    std::string deviceName_;
    uint8_t deviceUUID_[VK_UUID_SIZE] = {};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkPhysicalDeviceMemoryProperties memProps_{};
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;
    uint32_t descBindingCount_ = 0;
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
    std::map<std::string, VkPipeline> pipelines_;
    GpuBuffer staging_, stagingDl_;
    void* stagingPtr_ = nullptr;
    void* stagingDlPtr_ = nullptr;
    VkDeviceSize totalAllocated_ = 0;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> allocations_;  // 由上下文析构释放

    static constexpr uint32_t kMaxQueries = 16384;
    bool profiling_ = false;
    float timestampPeriod_ = 1.0f;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    std::vector<std::string> queryNames_;
    std::map<std::string, std::pair<uint64_t, double>> profStats_;  // 名称 ->（次数，总毫秒）
};
