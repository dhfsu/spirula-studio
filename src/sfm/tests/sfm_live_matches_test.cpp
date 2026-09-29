// 测试验证阶段持续追加匹配文件时的索引，覆盖未写完尾部及结束后切换到完整 matches.bin。
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sfm/core/Matches.h"
#include "sfm/core/Progress.h"
#include "sfm/tests/TestMain.h"

using namespace sfm;

static int fails = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static std::vector<FeatureMatch> matches(uint32_t n, uint32_t tag) {
    std::vector<FeatureMatch> m(n);
    for (uint32_t i = 0; i < n; i++) m[i] = {tag + i, tag + 2 * i, 0.0f};
    return m;
}

static int cmdLiveMatchesTest(int, char**) {
    const std::string dir = "sfm_live_matches_test.tmp";
    std::error_code rm;
    std::filesystem::remove_all(dir, rm);
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/live_matches.bin";

    progress::set_dir(dir);
    progress::live_matches_begin({"a", "b", "c"}, {10, 20, 30});
    const std::vector<std::vector<FeatureMatch>> pairs = {
        matches(4, 100), matches(7, 200), matches(2, 300)};
    const uint32_t ends[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int i = 0; i < 3; i++)
        progress::live_pair(ends[i][0], ends[i][1], 2, &pairs[i][0].idx1,
                            &pairs[i][0].idx2, sizeof(FeatureMatch),
                            (uint32_t)pairs[i].size());
    progress::flush();   // 写入器按时间刷新，不逐图像对刷新

    MatchesIndex idx;
    check(indexMatches(path, idx), "index a streaming file");
    check(idx.images.size() == 3, "image count survives");
    check(idx.images[2].name == "c" && idx.images[2].num_features == 30,
          "image entries survive");
    check(idx.pairs.size() == 3, "every complete pair is indexed");
    for (int i = 0; i < 3 && idx.pairs.size() == 3; i++) {
        std::vector<FeatureMatch> got;
        check(readPairMatches(path, idx.pairs[i], got), "read a pair back");
        check(got.size() == pairs[i].size(), "correspondence count round trips");
        bool same = got.size() == pairs[i].size();
        for (size_t k = 0; same && k < got.size(); k++)
            same = got[k].idx1 == pairs[i][k].idx1 && got[k].idx2 == pairs[i][k].idx2;
        check(same, "correspondences round trip");
    }

    // 任意截断均只返回完整记录前缀，不报错、不暴露未完成图像对。
    const uintmax_t full = std::filesystem::file_size(path);
    std::string bytes;
    {
        std::ifstream f(path, std::ios::binary);
        bytes.assign((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    }
    int torn_ok = 0;
    for (uintmax_t cut = 20; cut < full; cut++) {
        const std::string cut_path = dir + "/cut.bin";
        {
            std::ofstream f(cut_path, std::ios::binary | std::ios::trunc);
            f.write(bytes.data(), (std::streamsize)cut);
        }
        MatchesIndex t;
        if (!indexMatches(cut_path, t)) continue;   // 文件头尚未写出
        bool prefix = t.pairs.size() <= idx.pairs.size();
        for (size_t i = 0; prefix && i < t.pairs.size(); i++)
            prefix = t.pairs[i].image1 == idx.pairs[i].image1 &&
                     t.pairs[i].image2 == idx.pairs[i].image2 &&
                     t.pairs[i].count == idx.pairs[i].count;
        check(prefix, "a torn tail yields a prefix");
        if (!prefix) break;
        torn_ok++;
    }
    check(torn_ok > 0, "some truncations indexed at all");

    // 完整文件应报告真实对数，而非流式哨兵。
    MatchesDatabase db;
    db.images = {{"a", 10}, {"b", 20}};
    db.pairs.push_back({0, 1, 2, matches(3, 400)});
    const std::string done = dir + "/matches.bin";
    writeMatches(done, db);
    MatchesIndex fin;
    check(indexMatches(done, fin), "index a finished file");
    check(fin.pairs.size() == 1, "finished file indexes its pairs");

    // 固定对数文件截断仍须报错，不能当作流式前缀。
    {
        std::ifstream f(done, std::ios::binary);
        std::string b((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
        std::ofstream o(dir + "/short.bin", std::ios::binary | std::ios::trunc);
        o.write(b.data(), (std::streamsize)b.size() - 12);
    }
    MatchesIndex bad;
    check(!indexMatches(dir + "/short.bin", bad),
          "a truncated finished file is an error");

    progress::set_dir("");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::printf("live matches: 3 pairs, %d truncations indexed as prefixes\n", torn_ok);
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

int main() { return sfmTestMain(0, nullptr, cmdLiveMatchesTest); }
