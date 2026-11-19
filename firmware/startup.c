#include <stdint.h>

extern uint32_t _estack;
extern uint32_t _sidata, _sdata, _edata;
extern uint32_t _sbss, _ebss;

int main(void);
void Reset_Handler(void);
void Default_Handler(void);

/* Cortex-M exception vector table */
__attribute__((used, section(".isr_vector")))
const uintptr_t vector_table[] = {
    (uintptr_t)&_estack,
    (uintptr_t)Reset_Handler,
    (uintptr_t)Default_Handler, /* NMI */
    (uintptr_t)Default_Handler, /* HardFault */
    (uintptr_t)Default_Handler, /* MemManage */
    (uintptr_t)Default_Handler, /* BusFault */
    (uintptr_t)Default_Handler, /* UsageFault */
    0, 0, 0, 0,                 /* Reserved */
    (uintptr_t)Default_Handler, /* SVCall */
    (uintptr_t)Default_Handler, /* DebugMonitor */
    0,
    (uintptr_t)Default_Handler, /* PendSV */
    (uintptr_t)Default_Handler  /* SysTick */
};

/* Executed when the processor resets */
void Reset_Handler(void)
{
    /* Copy initialized data into RAM */
    uint32_t *src = &_sidata;

    for (uint32_t *dst = &_sdata; dst < &_edata;) {
        *dst++ = *src++;
    }

    /* Clear uninitialized variables */
    for (uint32_t *dst = &_sbss; dst < &_ebss;) {
        *dst++ = 0;
    }

    /* Start the application */
    (void)main();

    while (1) {
        __asm volatile("wfi");
    }
}

void Default_Handler(void)
{
    while (1) {
        __asm volatile("wfi");
    }
}
