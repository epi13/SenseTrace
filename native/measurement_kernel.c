#define _GNU_SOURCE

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <x86intrin.h>
#if defined(__linux__)
#include <sched.h>
#endif
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

/*
 * Versioned trajectory boundaries.  The historical v4 entry points above
 * deliberately keep their RDTSC/RDTSCP boundary semantics.  The v5
 * trajectory primitive uses RDTSCP at both ends so TSC_AUX is retained as an
 * endpoint observation.  This adds the RDTSCP overhead to every reported
 * duration; it is part of the v5 instrument definition.
 */
static inline uint64_t st_begin_trajectory(uint32_t *aux) {
    st_compiler_barrier();
    _mm_lfence();
    uint64_t value = __rdtscp(aux);
    _mm_lfence();
    st_compiler_barrier();
    return value;
}

static inline uint64_t st_end_trajectory(uint32_t *aux) {
    st_compiler_barrier();
    uint64_t value = __rdtscp(aux);
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

static uint64_t st_load_control(
    const volatile uint64_t *address, volatile uint64_t *sink, uint64_t delay_cycles
) {
    uint64_t started = st_begin();
    uint64_t value = *address;
    /*
     * The volatile access fixes the compiler-visible load.  LFENCE then
     * orders completion of prior loads before the artificial-delay clock
     * boundary on supported x86; it does not make a physical DRAM claim.
     */
    _mm_lfence();
    st_compiler_barrier();
    uint64_t delay_started = __rdtsc();
    st_compiler_barrier();
    if (delay_cycles > 0) {
        uint64_t deadline = delay_started + delay_cycles;
        while (__rdtsc() < deadline) {
            _mm_pause();
        }
    }
    uint64_t finished = st_end();
    *sink ^= value;
    return finished - started;
}

static uint64_t st_load_dependency_chain(
    const volatile uint64_t *address, volatile uint64_t *sink
) {
    uint64_t started = st_begin();
    uint64_t value = *address;
    /* Keep the loaded value live through a small, auditable dependency chain. */
    value ^= value << 7;
    value ^= value >> 9;
    value ^= value << 8;
    st_compiler_barrier();
    uint64_t finished = st_end();
    *sink ^= value;
    return finished - started;
}

static inline uint64_t st_trajectory_load(
    const volatile uint64_t *address,
    volatile uint64_t *sink,
    uint64_t delay_cycles,
    uint64_t *start_tsc,
    uint64_t *end_tsc,
    uint32_t *start_aux,
    uint32_t *end_aux
) {
    uint64_t started = st_begin_trajectory(start_aux);
    uint64_t value = *address;
    _mm_lfence();
    st_compiler_barrier();
    if (delay_cycles > 0) {
        uint64_t deadline = __rdtsc() + delay_cycles;
        while (__rdtsc() < deadline) {
            _mm_pause();
        }
    }
    uint64_t finished = st_end_trajectory(end_aux);
    *sink ^= value;
    *start_tsc = started;
    *end_tsc = finished;
    return finished - started;
}

static inline uint64_t st_trajectory_timer(
    uint64_t *start_tsc,
    uint64_t *end_tsc,
    uint32_t *start_aux,
    uint32_t *end_aux
) {
    uint64_t started = st_begin_trajectory(start_aux);
    uint64_t finished = st_end_trajectory(end_aux);
    *start_tsc = started;
    *end_tsc = finished;
    return finished - started;
}

static inline uint8_t st_trajectory_quality(uint32_t start_aux, uint32_t end_aux) {
    /* Bit 0: both endpoints are present; bit 1: endpoint identities differ. */
    return (uint8_t)(1u | (start_aux != end_aux ? 2u : 0u));
}

static inline void st_trajectory_evict(
    const volatile uint8_t *eviction,
    size_t eviction_bytes,
    volatile uint64_t *sink
) {
    if (eviction == NULL || eviction_bytes == 0) {
        return;
    }
    for (size_t index = 0; index < eviction_bytes; index += 64) {
        *sink ^= eviction[index];
    }
    st_compiler_barrier();
}
#endif

const char *st_kernel_version(void) {
    /* Historical v4 entry points retain their original ABI and semantics. */
    return "sensetrace-native-kernel-v4";
}

const char *st_trajectory_kernel_version(void) {
    return "sensetrace-native-trajectory-v1";
}

int st_cpu_supports_clflush(void) {
#if SENSETRACE_X86
    return st_has_clflush();
#else
    return 0;
#endif
}

int st_cpu_supports_rdtscp(void) {
#if SENSETRACE_X86
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(0x80000001u, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }
    return (edx & (1u << 27)) != 0;
#else
    return 0;
#endif
}

int st_cpu_supports_avx2(void) {
#if SENSETRACE_X86
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7u, 0u, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }
    return (ebx & (1u << 5)) != 0;
#else
    return 0;
#endif
}

int st_read_tsc_aux(uint64_t *tsc, uint32_t *aux) {
#if SENSETRACE_X86
    if (tsc == NULL || aux == NULL || !st_cpu_supports_rdtscp()) {
        return -EINVAL;
    }
    *tsc = st_begin_trajectory(aux);
    return 0;
#else
    (void)tsc;
    (void)aux;
    return -ENOTSUP;
#endif
}

int st_measure_cached_control(
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
        output[index] = st_load_control(address, &sink, delay_cycles);
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

int st_measure_flushed_control(
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
        output[index] = st_load_control(address, &sink, delay_cycles);
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

int st_measure_dependency_chain(
    const volatile uint64_t *address,
    size_t repetitions,
    uint64_t *output
) {
#if SENSETRACE_X86
    if (address == NULL || output == NULL || repetitions == 0) {
        return -EINVAL;
    }
    volatile uint64_t sink = *address;
    for (size_t index = 0; index < repetitions; ++index) {
        output[index] = st_load_dependency_chain(address, &sink);
    }
    return 0;
#else
    (void)address;
    (void)repetitions;
    (void)output;
    return -ENOTSUP;
#endif
}

int st_measure_repeated_load(
    const volatile uint64_t *address,
    size_t repetitions,
    uint64_t *output
) {
#if SENSETRACE_X86
    if (address == NULL || output == NULL || repetitions == 0) {
        return -EINVAL;
    }
    volatile uint64_t sink = *address;
    for (size_t index = 0; index < repetitions; ++index) {
        output[index] = st_load_control(address, &sink, 0);
    }
    return 0;
#else
    (void)address;
    (void)repetitions;
    (void)output;
    return -ENOTSUP;
#endif
}

int st_measure_paired_cached(
    const volatile uint64_t *first,
    const volatile uint64_t *second,
    size_t repetitions,
    int64_t *output
) {
#if SENSETRACE_X86
    if (first == NULL || second == NULL || output == NULL || repetitions == 0) {
        return -EINVAL;
    }
    volatile uint64_t sink = *first ^ *second;
    for (size_t index = 0; index < repetitions; ++index) {
        uint64_t first_ticks = st_load_control(first, &sink, 0);
        uint64_t second_ticks = st_load_control(second, &sink, 0);
        output[index] = (int64_t)second_ticks - (int64_t)first_ticks;
    }
    return 0;
#else
    (void)first;
    (void)second;
    (void)repetitions;
    (void)output;
    return -ENOTSUP;
#endif
}

/* Versioned batched trajectory conditions.  These values are part of the ABI. */
enum {
    ST_TRAJECTORY_CACHED_PRELOADED = 0,
    ST_TRAJECTORY_CLFLUSH = 1,
    ST_TRAJECTORY_EVICTION = 2,
    ST_TRAJECTORY_TIMER_ONLY = 3,
};

int st_measure_trajectory(
    const volatile uint64_t *target,
    const volatile uint64_t *reference,
    const volatile uint8_t *conditions,
    const volatile uint8_t *orders,
    size_t repetitions,
    const volatile uint8_t *eviction,
    size_t eviction_bytes,
    uint64_t *first_start_tsc,
    uint64_t *first_end_tsc,
    uint32_t *first_start_aux,
    uint32_t *first_end_aux,
    uint8_t *first_quality,
    uint64_t *second_start_tsc,
    uint64_t *second_end_tsc,
    uint32_t *second_start_aux,
    uint32_t *second_end_aux,
    uint8_t *second_quality
) {
#if SENSETRACE_X86
    if (target == NULL || conditions == NULL || repetitions == 0
        || first_start_tsc == NULL || first_end_tsc == NULL
        || first_start_aux == NULL || first_end_aux == NULL || first_quality == NULL) {
        return -EINVAL;
    }
    if (!st_cpu_supports_rdtscp()) {
        return -ENOTSUP;
    }
    if (eviction_bytes > 0 && eviction == NULL) {
        return -EINVAL;
    }
    for (size_t index = 0; index < repetitions; ++index) {
        if (conditions[index] > ST_TRAJECTORY_TIMER_ONLY) {
            return -EINVAL;
        }
        if (conditions[index] == ST_TRAJECTORY_CLFLUSH && !st_has_clflush()) {
            return -ENOTSUP;
        }
    }
    const int paired = reference != NULL;
    if (paired && (orders == NULL || second_start_tsc == NULL || second_end_tsc == NULL
        || second_start_aux == NULL || second_end_aux == NULL || second_quality == NULL)) {
        return -EINVAL;
    }

    volatile uint64_t sink = 0;
    /* Only the cached/preloaded condition is intentionally primed.  In
     * particular, eviction and timer-only rows do not read either target
     * before their measured interval. */
    int needs_preload = 0;
    for (size_t index = 0; index < repetitions; ++index) {
        if (conditions[index] == ST_TRAJECTORY_CACHED_PRELOADED) {
            needs_preload = 1;
            break;
        }
    }
    if (needs_preload) {
        sink ^= *target;
        if (paired) {
            sink ^= *reference;
        }
    }
    for (size_t index = 0; index < repetitions; ++index) {
        uint8_t condition = conditions[index];
        uint8_t order = paired ? (uint8_t)(orders[index] & 1u) : 0u;
        const volatile uint64_t *first_address = order == 0 ? target : reference;
        const volatile uint64_t *second_address = order == 0 ? reference : target;
        uint64_t *first_start = &first_start_tsc[index];
        uint64_t *first_end = &first_end_tsc[index];
        uint32_t *first_start_endpoint = &first_start_aux[index];
        uint32_t *first_end_endpoint = &first_end_aux[index];

        if (condition == ST_TRAJECTORY_CLFLUSH) {
            _mm_clflush((const void *)first_address);
            _mm_mfence();
        } else if (condition == ST_TRAJECTORY_EVICTION) {
            st_trajectory_evict(eviction, eviction_bytes, &sink);
        }
        if (condition == ST_TRAJECTORY_TIMER_ONLY) {
            st_trajectory_timer(first_start, first_end, first_start_endpoint, first_end_endpoint);
        } else {
            st_trajectory_load(first_address, &sink, 0, first_start, first_end,
                first_start_endpoint, first_end_endpoint);
        }
        first_quality[index] = st_trajectory_quality(
            first_start_aux[index], first_end_aux[index]);

        if (paired) {
            if (condition == ST_TRAJECTORY_CLFLUSH) {
                _mm_clflush((const void *)second_address);
                _mm_mfence();
            } else if (condition == ST_TRAJECTORY_EVICTION) {
                st_trajectory_evict(eviction, eviction_bytes, &sink);
            }
            if (condition == ST_TRAJECTORY_TIMER_ONLY) {
                st_trajectory_timer(&second_start_tsc[index], &second_end_tsc[index],
                    &second_start_aux[index], &second_end_aux[index]);
            } else {
                st_trajectory_load(second_address, &sink, 0, &second_start_tsc[index],
                    &second_end_tsc[index], &second_start_aux[index], &second_end_aux[index]);
            }
            second_quality[index] = st_trajectory_quality(
                second_start_aux[index], second_end_aux[index]);
        }
    }
    return 0;
#else
    (void)target;
    (void)reference;
    (void)conditions;
    (void)orders;
    (void)repetitions;
    (void)eviction;
    (void)eviction_bytes;
    (void)first_start_tsc;
    (void)first_end_tsc;
    (void)first_start_aux;
    (void)first_end_aux;
    (void)first_quality;
    (void)second_start_tsc;
    (void)second_end_tsc;
    (void)second_start_aux;
    (void)second_end_aux;
    (void)second_quality;
    return -ENOTSUP;
#endif
}

int st_run_memory_pressure(
    volatile uint64_t *buffer,
    size_t word_count,
    uint64_t duration_cycles,
    int operation,
    int requested_cpu,
    uint64_t *start_tsc,
    uint64_t *end_tsc,
    uint32_t *start_aux,
    uint32_t *end_aux,
    uint64_t *iterations,
    uint32_t *status,
    uint32_t *started
) {
#if SENSETRACE_X86 && defined(__linux__)
    if (buffer == NULL || word_count == 0 || duration_cycles == 0 || (operation != 0 && operation != 1)
        || start_tsc == NULL || end_tsc == NULL || start_aux == NULL || end_aux == NULL
        || iterations == NULL || status == NULL || started == NULL) {
        return -EINVAL;
    }
    cpu_set_t previous;
    int affinity_changed = 0;
    if (requested_cpu >= 0) {
        if (sched_getaffinity(0, sizeof(previous), &previous) != 0) {
            return -errno;
        }
        cpu_set_t requested;
        CPU_ZERO(&requested);
        CPU_SET(requested_cpu, &requested);
        if (sched_setaffinity(0, sizeof(requested), &requested) != 0) {
            return -errno;
        }
        affinity_changed = 1;
    }
    volatile uint64_t sink = 0;
    uint64_t completed = 0;
    *start_tsc = st_begin_trajectory(start_aux);
    *started = 1u;
    uint64_t deadline = *start_tsc + duration_cycles;
    while (__rdtsc() < deadline) {
        for (size_t index = 0; index < word_count; ++index) {
            if (operation == 0) {
                sink ^= buffer[index];
            } else {
                buffer[index] = sink + (uint64_t)index + completed;
            }
        }
        ++completed;
    }
    *end_tsc = st_end_trajectory(end_aux);
    *iterations = completed;
    *status = 0u;
    if (affinity_changed) {
        (void)sched_setaffinity(0, sizeof(previous), &previous);
    }
    return 0;
#else
    (void)buffer;
    (void)word_count;
    (void)duration_cycles;
    (void)operation;
    (void)requested_cpu;
    (void)start_tsc;
    (void)end_tsc;
    (void)start_aux;
    (void)end_aux;
    (void)iterations;
    (void)status;
    (void)started;
    return -ENOTSUP;
#endif
}

/* Preserve the original entry points for callers using the v2 ABI. */
int st_measure_cached(const volatile uint64_t *address, size_t repetitions, uint64_t *output) {
    return st_measure_cached_control(address, repetitions, 0, output);
}

int st_measure_flushed(const volatile uint64_t *address, size_t repetitions, uint64_t *output) {
    return st_measure_flushed_control(address, repetitions, 0, output);
}

int st_measure_cached_delayed(
    const volatile uint64_t *address,
    size_t repetitions,
    uint64_t delay_cycles,
    uint64_t *output
) {
    return st_measure_cached_control(address, repetitions, delay_cycles, output);
}

int st_measure_flushed_delayed(
    const volatile uint64_t *address,
    size_t repetitions,
    uint64_t delay_cycles,
    uint64_t *output
) {
    return st_measure_flushed_control(address, repetitions, delay_cycles, output);
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
