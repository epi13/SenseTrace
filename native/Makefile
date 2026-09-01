CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -fPIC
LDFLAGS ?= -shared

all: libsensetrace_measurement.so

libsensetrace_measurement.so: measurement_kernel.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

test: libsensetrace_measurement.so
	python -c "from sensetrace.acquisition.native import NativeMeasurementKernel; k=NativeMeasurementKernel.load(); assert k is not None and k.supports_clflush; print(k.provenance())"

clean:
	rm -f libsensetrace_measurement.so

.PHONY: all test clean
