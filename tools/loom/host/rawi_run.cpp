// rawi_run <model.gguf> <ids.txt> <logits.f32> <times.txt>
// ids.txt: one sentence per line, space-separated vocab ids. Writes every sentence's [T, 75] logits
// back to back as float32, and one wall time in microseconds per sentence (one warm-up call first).
#include "loom/loom.h"

#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: rawi_run model ids out times\n"); return 2; }
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    auto model = loom::GgufModel::load(argv[1], backend.get());
    loom::Session session(*model, backend.get());
    std::ifstream in(argv[2]);
    std::vector<std::vector<double>> sents;
    for (std::string line; std::getline(in, line);) {
        std::istringstream ss(line); std::vector<double> ids; double v;
        while (ss >> v) ids.push_back(v);
        sents.push_back(ids);
    }
    std::FILE* out = std::fopen(argv[3], "wb");
    std::FILE* times = std::fopen(argv[4], "w");
    session.bridge().call("infer", {{"tokens", sents[0]}});  // warm-up
    size_t n_out = 0;
    for (const auto& ids : sents) {
        auto t0 = std::chrono::steady_clock::now();
        auto res = std::get<std::vector<double>>(session.bridge().call("infer", {{"tokens", ids}}));
        auto t1 = std::chrono::steady_clock::now();
        if (res.size() != ids.size() * 75) {
            std::fprintf(stderr, "sentence %zu: %zu values, expected %zu\n", n_out, res.size(), ids.size() * 75);
            return 1;
        }
        std::vector<float> f(res.begin(), res.end());
        std::fwrite(f.data(), sizeof(float), f.size(), out);
        std::fprintf(times, "%lld\n", (long long)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        ++n_out;
    }
    std::fclose(out); std::fclose(times);
    std::fprintf(stderr, "in=%zu out=%zu\n", sents.size(), n_out);
    return 0;
}
