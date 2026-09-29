#pragma once

// 统一可执行程序的工具入口，由首参数或程序文件名选择；GUI 可重新运行自身作为子进程，避免依赖相邻工具文件。
// 各入口独立初始化；SS_BUILD_GUI 关闭时仅构建无显示、无 GL 依赖的命令行程序。
// 可用工具由构建设置的 SS_TOOL_* 宏决定。

#include <string>

namespace app {

// 实际调用名用于帮助与错误消息，如 spirula sfm 或符号链接名称；在各入口开始时由 argv[0] 设置一次。
inline std::string& program_name() {
    static std::string name = "spirula";
    return name;
}
inline void set_program_name(const char* argv0, const char* fallback) {
    program_name() = (argv0 && argv0[0]) ? argv0 : fallback;
}

// 把帮助文本中的示例程序名替换为实际调用名，使示例保持可读，避免逐条插入格式化占位符。
inline std::string help_text(const char* text, const char* written_as) {
    std::string s = text;
    const std::string& prog = program_name();
    const std::string was = written_as;
    if (prog == was || was.empty()) return s;
    for (size_t p = s.find(was); p != std::string::npos;
         p = s.find(was, p + prog.size()))
        s.replace(p, was.size(), prog);
    return s;
}

// 工具的子命令名称也用于匹配 argv[0]；spirula-sfm 符号链接等价于 spirula sfm。
constexpr const char* kToolTrain = "train";
constexpr const char* kToolMesh  = "mesh";
constexpr const char* kToolSfm   = "sfm";
constexpr const char* kToolSam   = "sam";
constexpr const char* kToolGeometry = "geometry";
constexpr const char* kToolGui   = "gui";
constexpr const char* kToolEncode = "encode";
constexpr const char* kToolPartition = "partition";

}  // 命名空间 app

#ifdef SS_TOOL_TRAIN
int spirula_train_main(int argc, char** argv);
#endif
#ifdef SS_TOOL_MESH
int spirula_mesh_main(int argc, char** argv);
#endif
#ifdef SS_TOOL_SFM
int spirula_sfm_main(int argc, char** argv);
#endif
#ifdef SS_TOOL_SAM
int spirula_sam_main(int argc, char** argv);
#endif
#ifdef SS_TOOL_GEOMETRY
int spirula_geometry_main(int argc, char** argv);
#endif
#ifdef SS_TOOL_GUI
int spirula_gui_main(int argc, char** argv);
#endif
#ifdef SS_TOOL_ENCODE
int spirula_encode_main(int argc, char** argv);
#endif
#ifdef SS_TOOL_PARTITION
int spirula_partition_main(int argc, char** argv);
#endif
