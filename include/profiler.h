#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

// Global accumulators for profiling data
extern std::unordered_map<std::string, double> total_time_ms;
extern std::unordered_map<std::string, int> call_count;

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

// Dump aggregated profiling results to a CSV file with columns:
// name,total_ms,call_count,avg_ms
void dump_profile_csv(const std::string& path);