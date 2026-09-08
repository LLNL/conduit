// Temporary sweep: finds the byte size at which a device forall copy beats the
// runtime memcpy in unified memory. Remove once DEVICE_COPY_MIN_BYTES is set.
//
// Paste everything between CONDUIT_COPY_SWEEP_BEGIN and CONDUIT_COPY_SWEEP_END
// back to the assistant.

#include "conduit.hpp"
#include "conduit_execution.hpp"
#include "conduit_memory_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "gtest/gtest.h"

#if defined(CONDUIT_USE_CUDA)
#include <cuda_runtime.h>
#define DEV_MEMCPY cudaMemcpy
#define DEV_SYNC cudaDeviceSynchronize
#define DEV_MALLOC cudaMalloc
#define DEV_FREE cudaFree
#define DEV_KIND(s, d) ((s) && (d) ? cudaMemcpyDeviceToDevice : \
                        (s) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice)
#elif defined(CONDUIT_USE_HIP)
#include <hip/hip_runtime.h>
#define DEV_MEMCPY hipMemcpy
#define DEV_SYNC hipDeviceSynchronize
#define DEV_MALLOC hipMalloc
#define DEV_FREE hipFree
#define DEV_KIND(s, d) ((s) && (d) ? hipMemcpyDeviceToDevice : \
                        (s) ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice)
#endif

using namespace conduit;
using namespace conduit::execution;

#if defined(CONDUIT_USE_DEVICE)
struct CopyWords
{
    const uint64 *src;
    uint64 *dst;
    CONDUIT_EXEC void operator()(index_t i) const { dst[i] = src[i]; }
};

static void runtime_copy(void *dst, const void *src, size_t n)
{
    const bool s = DeviceMemory::is_device_allocation(src);
    const bool d = DeviceMemory::is_device_allocation(dst);
    if (!s && !d)
    {
        memcpy(dst, src, n);
    }
    else
    {
        DEV_MEMCPY(dst, src, n, DEV_KIND(s, d));
    }
    DEV_SYNC();
}

static void kernel_copy(void *dst, const void *src, size_t n)
{
    ExecutionPolicy policy = ExecutionPolicy::device();
    forall(policy, 0, static_cast<int>(n / sizeof(uint64)),
           CopyWords{static_cast<const uint64*>(src), static_cast<uint64*>(dst)});
    CONDUIT_DEVICE_ERROR_CHECK(policy);
    DEV_SYNC();
}

template <typename F>
static double ms(F f)
{
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Buffers that bypass the umpire pools, so every allocation is untouched
// memory and the copy pays first-touch costs.
static void *fresh_alloc(bool device, size_t n)
{
    void *p = nullptr;
    if (device) DEV_MALLOC(&p, n); else p = malloc(n);
    return p;
}

static void fresh_free(bool device, void *p)
{
    if (device) DEV_FREE(p); else free(p);
}

static std::string read_line(const char *path)
{
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s.empty() ? "unknown" : s;
}

struct Dir { const char *name; bool src_dev; bool dst_dev; };
static const Dir dirs[] = {{"h2d", false, true},
                           {"d2h", true, false},
                           {"h2h", false, false},
                           {"d2d", true, true}};

static const size_t min_bytes = 64 * 1024;
static const size_t max_bytes = 64 * 1024 * 1024;
static const int reps = 5;

// Returns the smallest size from which the kernel wins at every larger size.
static size_t sweep(const char *mode, bool fresh, void *h_src, void *h_dst,
                    void *d_src, void *d_dst)
{
    size_t overall = 0;
    for (const Dir &dir : dirs)
    {
        size_t best = 0;
        for (size_t n = min_bytes; n <= max_bytes; n *= 2)
        {
            std::vector<double> tm, tk;
            for (int r = 0; r < reps; r++)
            {
                if (fresh)
                {
                    void *src = fresh_alloc(dir.src_dev, n);
                    void *dst = fresh_alloc(dir.dst_dev, n);
                    memset(src, 1, n);
                    tm.push_back(ms([&]() { runtime_copy(dst, src, n); }));
                    fresh_free(dir.src_dev, src);
                    fresh_free(dir.dst_dev, dst);

                    src = fresh_alloc(dir.src_dev, n);
                    dst = fresh_alloc(dir.dst_dev, n);
                    memset(src, 1, n);
                    tk.push_back(ms([&]() { kernel_copy(dst, src, n); }));
                    fresh_free(dir.src_dev, src);
                    fresh_free(dir.dst_dev, dst);
                }
                else
                {
                    void *src = dir.src_dev ? d_src : h_src;
                    void *dst = dir.dst_dev ? d_dst : h_dst;
                    if (r == 0)
                    {
                        runtime_copy(dst, src, n);
                        kernel_copy(dst, src, n);
                    }
                    tm.push_back(ms([&]() { runtime_copy(dst, src, n); }));
                    tk.push_back(ms([&]() { kernel_copy(dst, src, n); }));
                }
            }
            const double m = median(tm), k = median(tk);
            std::printf("mode=%s dir=%s bytes=%zu memcpy_ms=%.4f kernel_ms=%.4f winner=%s\n",
                        mode, dir.name, n, m, k, k < m ? "kernel" : "memcpy");
            if (k < m) { if (best == 0) best = n; } else best = 0;
        }
        std::printf("mode=%s dir=%s threshold=%zu\n", mode, dir.name, best);
        overall = std::max(overall, best);
    }
    std::printf("mode=%s recommend=%zu\n", mode, overall);
    return overall;
}
#endif

TEST(conduit_copy_threshold, sweep)
{
#if defined(CONDUIT_USE_DEVICE)
    if (!DeviceMemory::is_unified())
    {
        std::cout << "not unified memory, skipping" << std::endl;
        return;
    }

    void *h_src = HostMemory::allocate(max_bytes);
    void *h_dst = HostMemory::allocate(max_bytes);
    void *d_src = DeviceMemory::allocate(max_bytes);
    void *d_dst = DeviceMemory::allocate(max_bytes);
    memset(h_src, 1, max_bytes);
    memset(h_dst, 0, max_bytes);
    memset(d_src, 1, max_bytes);
    memset(d_dst, 0, max_bytes);

    const char *xnack = std::getenv("HSA_XNACK");
    std::printf("CONDUIT_COPY_SWEEP_BEGIN\n");
    std::printf("unified=1 hsa_xnack=%s thp=%s hugetlb_preload=%s reps=%d stat=median\n",
                xnack ? xnack : "unset",
                read_line("/sys/kernel/mm/transparent_hugepage/enabled").c_str(),
                std::getenv("LD_PRELOAD") ? std::getenv("LD_PRELOAD") : "none",
                reps);
    const size_t warm  = sweep("warm",  false, h_src, h_dst, d_src, d_dst);
    const size_t fresh = sweep("fresh", true,  h_src, h_dst, d_src, d_dst);
    std::printf("recommend warm=%zu fresh=%zu\n", warm, fresh);
    std::printf("CONDUIT_COPY_SWEEP_END\n");

    HostMemory::deallocate(h_src);
    HostMemory::deallocate(h_dst);
    DeviceMemory::deallocate(d_src);
    DeviceMemory::deallocate(d_dst);
#else
    std::cout << "no device support, skipping" << std::endl;
#endif
}
