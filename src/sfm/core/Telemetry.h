#pragma once
// 按内容识别视频中的 GPMF、Insta360 尾部、DJI djmd 或 CAMM 遥测，仅解析样本表与传感器字节，不读取图像。
// SensorTimeline 提供时间查询，SensorGauge 使用结果；sfm_telemetry_test 可打印诊断。

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sfm {

// 读取相机明确声明的投影，而非按图像形状猜测；GoPro PRJT 的 EACO/FSFB 与 PMOD 描述全景裁剪行及两侧填充。
struct VideoProjection {
    std::string name;
    std::vector<uint32_t> mode;
};

// 所有 t 均为容器时钟中相对首视频帧的秒数。
struct TelemetryVec {
    double t = 0;
    double x = 0, y = 0, z = 0;
};

struct TelemetryQuat {
    double t = 0;
    double w = 1, x = 0, y = 0, z = 0;
};

struct TelemetryGps {
    double t = 0;
    double unix_time = 0;      // 没有绝对时间戳时为 0
    double lat = 0, lon = 0;   // WGS-84，经纬度单位为度
    double alt = 0;            // 米，仅 has_alt 为真时读取
    bool has_alt = false;
    bool fix = false;          // 接收器自身确认定位有效
    double speed = -1;         // 地面速度 m/s，负值表示未知
    double track = -1;         // 从北顺时针的角度，负值表示未知
    double dop = 0;            // 精度衰减因子，0 表示未知
};

enum class TelemetryCarrier { None, Gpmf, Insta360, DjiDvtm, Camm };
const char* telemetry_carrier_name(TelemetryCarrier c);

struct Telemetry {
    TelemetryCarrier carrier = TelemetryCarrier::None;
    std::string camera, firmware, serial;
    VideoProjection projection;
    double video_duration = 0;   // 视频头中的时长，单位秒
    double video_fps = 0;        // 首个视频轨道的帧率，0 表示未知
    double video_unix_start = 0; // 0 表示未知
    double frame_readout = 0;    // 滚动快门读出时长，单位秒，0 表示未知

    // 传感器坐标系依厂商不同，已知约定写入 notes；GoPro 已按 ORIN 重排到 GPMF 规范坐标系。
    std::vector<TelemetryVec> gyro;         // rad/s
    std::vector<TelemetryVec> accel;        // m/s^2，包含重力
    std::vector<TelemetryVec> magnet;       // 微特斯拉
    std::vector<TelemetryVec> gravity;      // 传感器坐标系单位向量
    std::vector<TelemetryQuat> orientation; // 相机记录的融合姿态
    // 下列映射给出姿态作用的加速度分量顺序，实测 GoPro CORI 使用 ORIN 坐标的 (X,Z,Y)。
    std::string orientation_axes = "XYZ";
    std::vector<TelemetryGps> gps;
    std::vector<std::string> notes;

    bool empty() const {
        return gyro.empty() && accel.empty() && gravity.empty() &&
               orientation.empty() && gps.empty();
    }
};

// 无法打开或不支持的容器返回 false 并设置 error；容器没有遥测不算错误，carrier 为 None 且输出为空。
bool telemetry_read(const std::string& path, Telemetry& out, std::string& error);

bool telemetry_read(const uint8_t* data, size_t size, Telemetry& out, std::string& error);

// 使用调用方的随机读取接口，read 从 off 填充 n 字节，越界返回 false；适用于大文件和浏览器 File API。
using TelemetryRead = std::function<bool(uint64_t off, void* dst, size_t n)>;
bool telemetry_read(uint64_t size, const TelemetryRead& read, Telemetry& out, std::string& error);

// 仅从容器盒读取投影信息，不读取传感器样本；未声明投影时为空。
VideoProjection video_projection(const std::string& path);

// 检查传感器数据是否基本可用，包括单位、视频覆盖、采样规律、重力模长及 GPS 是否更新，不代表精度保证。
struct TelemetryStreamCheck {
    size_t count = 0;
    double t_first = 0, t_last = 0;
    double rate_hz = 0;        // 采样率为时间间隔中位数的倒数
    double max_gap = 0;        // 秒
    size_t non_monotonic = 0;
    size_t non_finite = 0;
};

struct TelemetryCheck {
    TelemetryStreamCheck gyro, accel, gravity, orientation, gps;
    double accel_norm_median = 0;  // 半秒平均后的 m/s^2，单位正确时约 9.81
    double accel_norm_spread = 0;  // |原始值 - 平均值| 的中位数，表示振动
    double gyro_norm_median = 0;   // rad/s
    double orientation_norm_err = 0;  // max | |q| - 1 |
    double orientation_max_step_deg = 0;
    // 不同流的重力方向一致性，单位度，无法计算时为负。
    double grav_vs_accel_deg = -1;        // GRAV 与加速度计的比较，均在传感器坐标系
    double accel_in_world_spread_deg = -1;  // 整段采集中经姿态旋转的加速度
    bool attitude_is_sensor_to_world = false;  // 使重力方向恒定的四元数变换方向
    double gps_fix_fraction = 0;
    double gps_frozen_fraction = 0;  // 位置逐位相同的连续样本数
    size_t gps_distinct = 0;         // 位置不同于前一样本的数量
    double gps_longest_hold = 0;     // 同一位置持续重复的秒数
    size_t gps_outliers = 0;         // 因 DOP > 10 或相对上个保留定位速度 > 50 m/s 而丢弃的定位数
    double gps_spread_m = 0;         // 保留定位相对均值的 RMS 半径
    double gps_path_m = 0;           // 相邻保留定位间距离之和
    double gps_speed_max = 0;        // 接收器报告的速度，m/s
    bool imu_usable = false;
    bool gps_usable = false;
    std::vector<std::string> warnings;
};

TelemetryCheck telemetry_check(const Telemetry& t);

// 保留有效、DOP <= 10 且相对上个不同位置不超过 50 m/s 的定位，返回后两项筛选丢弃数；MAX 曾出现自身 DOP 合格但明显异常的定位。
size_t telemetry_gps_filter(const Telemetry& t, std::vector<TelemetryGps>& kept);
bool gps_valid(const TelemetryGps& g);
double telemetry_haversine_m(double lat1, double lon1, double lat2, double lon2);

// sfm_telemetry_test FILE 输出的开发诊断表，固定使用英文。
std::string telemetry_report(const Telemetry& t, const TelemetryCheck& c);

}  // 命名空间 sfm
