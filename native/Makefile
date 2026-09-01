CC ?= cc
PYTHON ?= python3
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -fPIC
LDFLAGS ?= -shared

all: libsensetrace_measurement.so

libsensetrace_measurement.so: measurement_kernel.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

test: libsensetrace_measurement.so
	$(PYTHON) -c "import ctypes; lib = ctypes.CDLL('./libsensetrace_measurement.so'); lib.st_cpu_supports_clflush.restype = ctypes.c_int; lib.st_kernel_version.restype = ctypes.c_char_p; assert lib.st_cpu_supports_clflush(); print(lib.st_kernel_version().decode('ascii'))"

clean:
	rm -f libsensetrace_measurement.so

.PHONY: all test clean
