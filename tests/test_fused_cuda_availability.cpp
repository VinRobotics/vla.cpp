// Verify that unsupported devices cannot emit fused CUSTOM nodes, even when
// the environment explicitly requests fusion.
#include "cuda/vla_cuda_ops.h"
#include <cuda_runtime_api.h>
#include <cstdio>

static void fusion_env(const char * value) {
#ifdef _WIN32
    _putenv_s("VLA_FUSED_OPS", value ? value : "");
#else
    if (value) setenv("VLA_FUSED_OPS", value, 1);
    else unsetenv("VLA_FUSED_OPS");
#endif
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::puts("fused_cuda_availability: no CUDA device, skipping");
        return 0;
    }
    fusion_env(nullptr);
    const bool defaults = vla::enable_fused_ops("test", true);
    if (prop.major < 8 && (defaults || vla::fused_ops_active())) return 1;
    fusion_env("1");
    const bool requested = vla::enable_fused_ops("test", true);
    if (requested != defaults) return 2;
    if (prop.major < 8 && vla::fused_ops_active()) return 3;
    fusion_env("0");
    if (vla::enable_fused_ops("test", true)) return 4;
    fusion_env("1");
    if (vla::enable_fused_ops("test", false)) return 5;
    fusion_env(nullptr);
    std::puts("fused_cuda_availability: PASS");
    return 0;
}
