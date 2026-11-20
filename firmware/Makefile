
.RECIPEPREFIX := >

# ============================================================
# Cross toolchain: bare-metal ARM Cortex-M4 firmware
# ============================================================

CC = arm-none-eabi-gcc

CFLAGS = -std=c11 \
         -mcpu=cortex-m4 \
         -mthumb \
         -Os -g3 \
         -Wall -Wextra \
         -ffreestanding \
         -fno-builtin \
         -ffunction-sections \
         -fdata-sections \
         -fstack-usage

LDFLAGS = -nostdlib \
          -T linker.ld \
          -Wl,--gc-sections \
          -Wl,-Map=build/tinyguard.map

SOURCES = startup.c main.c sensor_sim.c signal_features.c \
          tinyguard_model.c tinyguard_int8.c

HEADERS = sensor_sim.h signal_features.h \
          tinyguard_model.h tinyguard_int8.h \
          model_params.h model_int8_params.h

# ============================================================
# Host toolchain: verification binaries driven by the ml/ scripts
# ============================================================
#
# These four programs run the SAME inference C sources natively so the Python
# reference implementation can be compared against the exact C arithmetic.
# ml/verify_int8_c.py and ml/verify_embedded_model.py execute them, so
# `make hosttools` is a prerequisite for those scripts.

HOSTCC = gcc

HOSTCFLAGS = -std=c11 -O2 -Wall -Wextra

HOSTTOOLS = verify_int8_host \
            verify_model_host \
            test_model_host \
            debug_int8_host

all: build/tinyguard.elf

build:
>mkdir -p build

build/tinyguard.elf: $(SOURCES) $(HEADERS) linker.ld | build
>$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) -lgcc -o $@

run: build/tinyguard.elf
>qemu-system-arm -M mps2-an386 \
> -nographic \
> -semihosting-config enable=on,target=native \
> -kernel build/tinyguard.elf

size: build/tinyguard.elf
>arm-none-eabi-size build/tinyguard.elf

disasm: build/tinyguard.elf
>arm-none-eabi-objdump -d build/tinyguard.elf

# ---- Host verification binaries ----

hosttools: $(HOSTTOOLS)

# INT8 engine: reads "mean rms peak" per line, writes "logit prediction".
verify_int8_host: verify_int8_host.c tinyguard_int8.c tinyguard_int8.h model_int8_params.h
>$(HOSTCC) $(HOSTCFLAGS) verify_int8_host.c tinyguard_int8.c -o $@

# FP32 engine: reads "mean,rms,peak" per line, writes "logit,prediction".
verify_model_host: verify_model_host.c tinyguard_model.c tinyguard_model.h model_params.h
>$(HOSTCC) $(HOSTCFLAGS) verify_model_host.c tinyguard_model.c -o $@

# FP32 smoke test on two fixed windows.
test_model_host: test_model_host.c tinyguard_model.c tinyguard_model.h model_params.h
>$(HOSTCC) $(HOSTCFLAGS) test_model_host.c tinyguard_model.c -o $@

# INT8 layer-by-layer dump for one window, for debugging quantization.
debug_int8_host: debug_int8_host.c model_int8_params.h
>$(HOSTCC) $(HOSTCFLAGS) debug_int8_host.c -o $@

# ---- Verification pipeline ----

# Build the host tools, then run the Python cross-checks against them.
verify: hosttools
>cd ../ml && python verify_embedded_model.py
>cd ../ml && python verify_int8_c.py

clean:
>rm -rf build
>rm -f $(HOSTTOOLS) $(addsuffix .exe,$(HOSTTOOLS))
>rm -f *.su

.PHONY: all run size disasm clean hosttools verify
