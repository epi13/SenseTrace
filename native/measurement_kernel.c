#define _GNU_SOURCE

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <x86intrin.h>
#define SENSETRACE_X86 1
#else
#define SENSETRACE_X86 0
#endif

#if SENSETRACE_X86
static inline void st_compiler_barrier(void) {
    __asm__ __volatile__("" ::: "memory");
}

static inline uint64_t st_begin(void) {
    st_compiler_barrier();
    _mm_lfence();
    uint64_t value = __rdtsc();
    st_compiler_barrier();
    return value;
}

static inline uint64_t st_end(void) {
    unsigned int aux = 0;
    st_compiler_barrier();
    uint64_t value = __rdtscp(&aux);
    _mm_lfence();
    st_compiler_barrier();
    return value;
}

static int st_has_clflush(void) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }
    return (edx & (1u << 19)) != 0;
}

static uint64_t st_load(const volatile uint64_t *address, volatile uint64_t *sink) {
    uint64_t started = st_begin();
    uint64_t value = *address;
    uint64_t finished = st_end();
    *sink ^= value;
    return finished - started;
}

static uint64_t st_load_delayed(
    const volatile uint64_t *address, volatile uint64_t *sink, uint64_t delay_cycles
) {
    uint64_t started = st_begin();
    uint64_t value = *address;
    st_compiler_barrier();
    if (delay_cycles > 0) {
        uint64_t deadline = __rdtsc() + delay_cycles;
        while (__rdtsc() < deadline) {
            _mm_pause();
        }
    }
    uint64_t finished = st_end();
    *sink ^= value;
    return finished - started;
}
#endif

const char *st_kernel_version(void) {
    return "sensetrace-native-kernel-v2";
}

int st_cpu_supports_clflush(void) {
#if SENSETRACE_X86
    return st_has_clflush();
#else
    return 0;
#endif
}

int st_measure_cached(const volatile uint64_t *address, size_t repetitions, uint64_t *output) {
#if SENSETRACE_X86
    if (address == NULL || output == NULL || repetitions == 0) {
        return -EINVAL;
    }
    volatile uint64_t sink = *address;
    for (size_t index = 0; index < repetitions; ++index) {
        output[index] = st_load(address, &sink);
    }
    return 0;
#else
    (void)address;
    (void)repetitions;
    (void)output;
    return -ENOTSUP;
#endif
}

int st_measure_flushed(const volatile uint64_t *address, size_t repetitions, uint64_t *output) {
#if SENSETRACE_X86
    if (address == NULL || output == NULL || repetitions == 0) {
        return -EINVAL;
    }
    if (!st_has_clflush()) {
        return -ENOTSUP;
    }
    volatile uint64_t sink = 0;
    for (size_t index = 0; index < repetitions; ++index) {
        _mm_clflush((const void *)address);
        _mm_mfence();
        output[index] = st_load(address, &sink);
    }
    return 0;
#else
    (void)address;
    (void)repetitions;
    (void)output;
    return -ENOTSUP;
#endif
}

int st_measure_cached_delayed(
    const volatile uint64_t *address,
    size_t repetitions,
    uint64_t delay_cycles,
    uint64_t *output
) {
#if SENSETRACE_X86
    if (address == NULL || output == NULL || repetitions == 0) {
        return -EINVAL;
    }
    volatile uint64_t sink = *address;
    for (size_t index = 0; index < repetitions; ++index) {
        output[index] = st_load_delayed(address, &sink, delay_cycles);
    }
    return 0;
#else
    (void)address;
    (void)repetitions;
    (void)delay_cycles;
    (void)output;
    return -ENOTSUP;
#endif
}

int st_measure_flushed_delayed(
    const volatile uint64_t *address,
    size_t repetitions,
    uint64_t delay_cycles,
    uint64_t *output
) {
#if SENSETRACE_X86
    if (address == NULL || output == NULL || repetitions == 0) {
        return -EINVAL;
    }
    if (!st_has_clflush()) {
        return -ENOTSUP;
    }
    volatile uint64_t sink = 0;
    for (size_t index = 0; index < repetitions; ++index) {
        _mm_clflush((const void *)address);
        _mm_mfence();
        output[index] = st_load_delayed(address, &sink, delay_cycles);
    }
    return 0;
#else
    (void)address;
    (void)repetitions;
    (void)delay_cycles;
    (void)output;
    return -ENOTSUP;
#endif
}

int st_timer_calibration(size_t repetitions, uint64_t *output) {
#if SENSETRACE_X86
    if (output == NULL || repetitions == 0) {
        return -EINVAL;
    }
    for (size_t index = 0; index < repetitions; ++index) {
        uint64_t started = st_begin();
        uint64_t finished = st_end();
        output[index] = finished - started;
    }
    return 0;
#else
    (void)repetitions;
    (void)output;
    return -ENOTSUP;
#endif
}

int st_idle_calibration(size_t repetitions, uint64_t *output) {
    return st_timer_calibration(repetitions, output);
}
