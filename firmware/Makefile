
.RECIPEPREFIX := >

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

all: build/tinyguard.elf

build:
>mkdir -p build

build/tinyguard.elf: $(SOURCES) sensor_sim.h signal_features.h \
                     tinyguard_model.h tinyguard_int8.h \
                     model_params.h model_int8_params.h linker.ld | build
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

clean:
>rm -rf build

.PHONY: all run size disasm clean
