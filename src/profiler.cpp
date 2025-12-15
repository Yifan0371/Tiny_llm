//
// Created by liyif on 2025/12/15.
//
#include "profiler.h"

#include <fstream>
#include <mutex>

std::unordered_map<std::string, double> total_time_ms;
std::unordered_map<std::string, int> call_count;

namespace {
std::mutex g_profiler_mutex;
}

ScopedTimer::ScopedTimer(const std::string& name)
    : name_(name), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start_).count();

    std::lock_guard<std::mutex> lock(g_profiler_mutex);
    total_time_ms[name_] += elapsed_ms;
    call_count[name_] += 1;
}

void dump_profile_csv(const std::string& path) {
    std::lock_guard<std::mutex> lock(g_profiler_mutex);
    std::ofstream fout(path, std::ios::out | std::ios::trunc);
    if (!fout) {
        return;
    }

    fout << "name,total_ms,call_count,avg_ms\n";
    for (const auto& entry : total_time_ms) {
        const std::string& name = entry.first;
        double total_ms = entry.second;
        int calls = call_count[name];
        double avg_ms = (calls > 0) ? (total_ms / calls) : 0.0;
        fout << name << ',' << total_ms << ',' << calls << ',' << avg_ms << "\n";
    }
}