/* Bare-metal RV32 runtime for the wasm2c build of fx-core.wasm.
 *
 * Provides what wasm2c's output needs from wasm-rt (linear memory, the funcref
 * table, traps) plus memcpy/memmove/memset/memcmp, and the entry point. The wasm
 * imports are ecalls (imports.c). Memory map (see link.ld):
 *   0x00010000  code, rodata, data, bss
 *   0x01FFFFF0  C stack top (grows down)
 *   0x02000000  wasm linear memory (up to 32 MB)
 */
#include <stddef.h>
#include <stdint.h>
#include "fxcore.h"

#define LINEAR_BASE ((uint8_t*)0x02000000u)
#define LINEAR_MAX_PAGES 512u /* 32 MB */
#define TABLE_MAX 8192u

volatile uint32_t rv_linear_bytes; /* read by the host for peak memory */

static void sys_exit(uint32_t code, uint32_t nr) {
  register uint32_t a0 asm("a0") = code;
  register uint32_t a7 asm("a7") = nr;
  asm volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
  for (;;) {
  }
}

void wasm_rt_init(void) {}
bool wasm_rt_is_initialized(void) { return true; }
void wasm_rt_free(void) {}

void wasm_rt_trap(wasm_rt_trap_t code) { sys_exit(1000u + (uint32_t)code, 94); __builtin_unreachable(); }

void wasm_rt_allocate_memory(wasm_rt_memory_t* m, uint64_t initial, uint64_t max, bool is64, uint32_t page_size) {
  if (initial > LINEAR_MAX_PAGES || page_size != 65536u) wasm_rt_trap(WASM_RT_TRAP_OOB);
  m->data = LINEAR_BASE;
  m->page_size = page_size;
  m->pages = initial;
  m->max_pages = max < LINEAR_MAX_PAGES ? max : LINEAR_MAX_PAGES;
  m->size = initial * 65536u;
  m->data_end = m->data + m->size;
  m->is64 = is64;
  rv_linear_bytes = (uint32_t)m->size;
}

uint64_t wasm_rt_grow_memory(wasm_rt_memory_t* m, uint64_t delta) {
  uint64_t old = m->pages;
  if (old + delta > m->max_pages) return (uint64_t)-1;
  m->pages = old + delta;
  m->size = m->pages * 65536u;
  m->data_end = m->data + m->size;
  rv_linear_bytes = (uint32_t)m->size;
  return old;
}

void wasm_rt_free_memory(wasm_rt_memory_t* m) { (void)m; }

static wasm_rt_funcref_t table_store[TABLE_MAX];

void wasm_rt_allocate_funcref_table(wasm_rt_funcref_table_t* t, uint32_t elements, uint32_t max) {
  if (elements > TABLE_MAX) wasm_rt_trap(WASM_RT_TRAP_OOB);
  t->data = table_store;
  t->size = elements;
  t->max_size = max < TABLE_MAX ? max : TABLE_MAX;
}

void wasm_rt_free_funcref_table(wasm_rt_funcref_table_t* t) { (void)t; }

/* ---- libc pieces (this file is built with -ffreestanding -fno-builtin) ---- */
void* memcpy(void* restrict d, const void* restrict s, size_t n) {
  uint8_t* dp = d;
  const uint8_t* sp = s;
  if ((((uintptr_t)dp ^ (uintptr_t)sp) & 3) == 0) {
    while (n && ((uintptr_t)dp & 3)) {
      *dp++ = *sp++;
      n--;
    }
    uint32_t* dw = (uint32_t*)dp;
    const uint32_t* sw = (const uint32_t*)sp;
    while (n >= 16) {
      uint32_t a = sw[0], b = sw[1], c = sw[2], e = sw[3];
      dw[0] = a;
      dw[1] = b;
      dw[2] = c;
      dw[3] = e;
      dw += 4;
      sw += 4;
      n -= 16;
    }
    while (n >= 4) {
      *dw++ = *sw++;
      n -= 4;
    }
    dp = (uint8_t*)dw;
    sp = (const uint8_t*)sw;
  }
  while (n--) *dp++ = *sp++;
  return d;
}

void* memmove(void* d, const void* s, size_t n) {
  uint8_t* dp = d;
  const uint8_t* sp = s;
  if (dp == sp || n == 0) return d;
  if (dp < sp || dp >= sp + n) return memcpy(d, s, n);
  dp += n;
  sp += n;
  if ((((uintptr_t)dp ^ (uintptr_t)sp) & 3) == 0) {
    while (n && ((uintptr_t)dp & 3)) {
      *--dp = *--sp;
      n--;
    }
    while (n >= 4) {
      dp -= 4;
      sp -= 4;
      *(uint32_t*)dp = *(const uint32_t*)sp;
      n -= 4;
    }
  }
  while (n--) *--dp = *--sp;
  return d;
}

void* memset(void* d, int c, size_t n) {
  uint8_t* dp = d;
  uint32_t w = (uint8_t)c * 0x01010101u;
  while (n && ((uintptr_t)dp & 3)) {
    *dp++ = (uint8_t)c;
    n--;
  }
  uint32_t* dw = (uint32_t*)dp;
  while (n >= 16) {
    dw[0] = w;
    dw[1] = w;
    dw[2] = w;
    dw[3] = w;
    dw += 4;
    n -= 16;
  }
  while (n >= 4) {
    *dw++ = w;
    n -= 4;
  }
  dp = (uint8_t*)dw;
  while (n--) *dp++ = (uint8_t)c;
  return d;
}

int memcmp(const void* a, const void* b, size_t n) {
  const uint8_t* x = a;
  const uint8_t* y = b;
  for (; n; n--, x++, y++)
    if (*x != *y) return *x - *y;
  return 0;
}

int bcmp(const void* a, const void* b, size_t n) { return memcmp(a, b, n); }

size_t strlen(const char* s) {
  const char* p = s;
  while (*p) p++;
  return (size_t)(p - s);
}

size_t wcslen(const int* s) {
  const int* p = s;
  while (*p) p++;
  return (size_t)(p - s);
}

static w2c_fxcore instance;

int rv_main(void) {
  wasm_rt_init();
  wasm2c_fxcore_instantiate(&instance, NULL, NULL);
  { register uint32_t a7 asm("a7") = 95; asm volatile("ecall" : : "r"(a7) : "memory"); } /* phase marker */
  w2c_fxcore_0x5Fstart(&instance);
  return 0;
}

__attribute__((naked, section(".text.start"))) void _start(void) {
  asm volatile(
      "li sp, 0x01FFFFF0\n"
      "call rv_main\n"
      "li a7, 93\n"
      "ecall\n"
      "1: j 1b\n");
}
