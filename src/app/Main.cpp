// 统一程序入口，工具定义见 app/Tools.h。
// spirula 或 spirula <file-or-folder> 打开窗口；sfm、train、sam、geometry、mesh 分别执行重建、训练、分割、几何估计和网格提取。
// 首参数若不是子命令则交给 GUI；无 GUI 构建会报告可用子命令。

#include "app/AppPaths.h"
#include "app/CrashLog.h"
#include "app/Tools.h"
#include "i18n/Locale.h"
#include "i18n/catalog/Cli.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <io.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>   // NOMINMAX 由 CMakeLists.txt 定义
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace cmsg = spirula::i18n::msg::cli;

#ifdef _WIN32
UINT g_console_cp = 0;

// Windows 控制台代码页独立于程序清单；临时切换为 UTF-8 以正确输出中文路径，退出时恢复，避免影响 cmd.exe。
void use_utf8_console() {
    const UINT prev = GetConsoleOutputCP();
    if (prev == 0 || prev == CP_UTF8 || !SetConsoleOutputCP(CP_UTF8)) return;
    g_console_cp = prev;
    std::atexit([] { SetConsoleOutputCP(g_console_cp); });
}
#endif

// 子命令名称是固定标识符，简介则通过 Msg 随 --lang 翻译；保存 Msg 指针而非缓存字符串，避免语言切换后引用失效。
struct Tool {
    const char* name;
    const spirula::i18n::Msg* summary;
    int (*run)(int, char**);
};

const std::vector<Tool>& tools() {
    static const std::vector<Tool> kTools = {
#ifdef SS_TOOL_GUI
        {app::kToolGui, &cmsg::tool_gui, spirula_gui_main},
#endif
#ifdef SS_TOOL_SFM
        {app::kToolSfm, &cmsg::tool_sfm, spirula_sfm_main},
#endif
#ifdef SS_TOOL_TRAIN
        {app::kToolTrain, &cmsg::tool_train, spirula_train_main},
#endif
#ifdef SS_TOOL_SAM
        {app::kToolSam, &cmsg::tool_sam, spirula_sam_main},
#endif
#ifdef SS_TOOL_GEOMETRY
        {app::kToolGeometry, &cmsg::tool_geometry, spirula_geometry_main},
#endif
#ifdef SS_TOOL_MESH
        {app::kToolMesh, &cmsg::tool_mesh, spirula_mesh_main},
#endif
#ifdef SS_TOOL_ENCODE
        {app::kToolEncode, &cmsg::tool_encode, spirula_encode_main},
#endif
#ifdef SS_TOOL_PARTITION
        {app::kToolPartition, &cmsg::tool_partition, spirula_partition_main},
#endif
    };
    return kTools;
}

const Tool* find_tool(const char* name) {
    if (!name) return nullptr;
    for (const Tool& t : tools())
        if (std::strcmp(t.name, name) == 0) return &t;
    return nullptr;
}

// 按 argv[0] 的文件名识别工具，spirula-sfm 的副本或符号链接均可启动 SfM；仅检查前缀后的名称，并兼容 ssplat- 前缀。
const Tool* tool_from_argv0(const char* argv0) {
    if (!argv0) return nullptr;
    std::string s = argv0;
    const size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos) s = s.substr(slash + 1);
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    if (s.size() > 4 && s.compare(s.size() - 4, 4, ".exe") == 0)
        s.resize(s.size() - 4);
    for (const char* prefix : {"spirula-", "ssplat-"})
        if (s.rfind(prefix, 0) == 0)
            return find_tool(s.c_str() + std::strlen(prefix));
    return nullptr;
}

void print_usage() {
    std::printf("Spirula Studio " SS_VERSION "\n\n");
#ifdef SS_TOOL_GUI
    std::printf("  spirula                        %s\n",
                cmsg::usage_open_app.get());
    std::printf("  spirula <file-or-folder>       %s\n",
                cmsg::usage_open_target.get());
#endif
    std::printf("  spirula <command> [options]\n\n");
    std::printf("%s\n", cmsg::usage_commands.get());
    for (const Tool& t : tools())
        std::printf("  %-8s %s\n", t.name, t.summary->get());
    std::printf("\n%s\n", cmsg::usage_per_command_help.get());
    std::printf("\n%s\n\n%s", cmsg::usage_lang.get(),
                spirula::i18n::language_list().c_str());
}

}  // 匿名命名空间

int main(int argc, char** argv) {
    // 输出到管道时禁用全缓冲，避免 4 KB 缓冲积压进度，让子进程看似卡住。
    // Windows CRT 将 _IOLBF 当作 _IOFBF，必须使用 _IONBF；其缓冲模式不接受大小 0，否则触发 __fastfail，以 0xC0000409 退出且无任何输出。
    if (!isatty(fileno(stdout))) {
#ifdef _WIN32
        std::setvbuf(stdout, nullptr, _IONBF, 0);
#else
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
#endif
    }
#ifdef _WIN32
    use_utf8_console();
#endif

    // 在统一入口处理 --lang 并从 argv 删除；语言选择优先级见 src/i18n/Locale.h，GUI 设置的优先级低于命令行和环境变量。
    const char* lang = spirula::i18n::take_lang_arg(&argc, argv);
    spirula::i18n::init(lang, nullptr);

    // 所有工具都安装崩溃报告，便于诊断仅向 GUI 父进程返回退出码的重建、分割和网格子进程。
    app::install_crash_log(app::config_dir());

    // 显式子命令优先于 argv[0] 的工具提示；spirula-sfm auto 中 auto 不是顶层子命令，因此继续按程序名称分派到 SfM。
    if (argc > 1) {
        if (const Tool* t = find_tool(argv[1])) {
            // 将 spirula sfm 作为工具程序名，使帮助示例可直接复制执行。
            std::string prog = std::string(argv[0] ? argv[0] : "spirula") + " " + t->name;
            app::set_crash_note(prog);
            std::vector<char*> sub;
            sub.push_back(prog.data());
            for (int i = 2; i < argc; i++) sub.push_back(argv[i]);
            sub.push_back(nullptr);
            return t->run((int)sub.size() - 1, sub.data());
        }
    }

    // 按工具名启动时完整传递参数，包括 --help 和 --version，使符号链接与独立程序行为一致。
    if (const Tool* t = tool_from_argv0(argc > 0 ? argv[0] : nullptr)) {
        app::set_crash_note(t->name);
        return t->run(argc, argv);
    }

    if (argc > 1) {
        const std::string a = argv[1];
        if (a == "help" || a == "--help" || a == "-h") {
            print_usage();
            return 0;
        }
        if (a == "--version" || a == "-v") {
            std::printf("%s\n", SS_VERSION);
            return 0;
        }
    }

#ifdef SS_TOOL_GUI
    return spirula_gui_main(argc, argv);
#else
    if (argc > 1)
        std::fprintf(stderr, "%s\n\n",
                     spirula::i18n::format(cmsg::err_unknown_command,
                                           {argv[1]}).c_str());
    else
        std::fprintf(stderr, "%s\n\n", cmsg::err_no_gui.get());
    print_usage();
    return 2;
#endif
}
