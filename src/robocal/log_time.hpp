#pragma once
// Stage timing for the [robocal] log: " [12.3 s, t=845 s]" = time since the previous stamped
// line (the stage that just finished) and since program start. Main thread only.
#include <chrono>
#include <cstdio>
#include <string>

namespace robocal {

inline std::chrono::steady_clock::time_point& LogClockStart() {
    static std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    return t0;
}

inline std::string Took() {
    using C = std::chrono::steady_clock;
    static C::time_point last = LogClockStart();
    const C::time_point now = C::now();
    char b[64];
    std::snprintf(b, sizeof b, " [%.1f s, t=%.0f s]", std::chrono::duration<double>(now - last).count(),
                  std::chrono::duration<double>(now - LogClockStart()).count());
    last = now;
    return b;
}

}  // namespace robocal
