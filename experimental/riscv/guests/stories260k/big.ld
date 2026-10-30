/* The chipformer RV32 address map (P2): RAM at 0 (data / TLS / bss / heap from 0x100, the stack
   below 0x8000), read-only data in the DROM at 0x20000 (128 KB), code in the ROM at 0x40000
   (256 KB). crt0 points tp at __tls_base (the single thread's TLS block: .tdata then .tbss). */
ENTRY(_start)
MEMORY
{
  RAM (rw) : ORIGIN = 0x100, LENGTH = 0x7F00
  DROM (r) : ORIGIN = 0x100000, LENGTH = 1024K
  ROM (rx) : ORIGIN = 0x40000, LENGTH = 256K
}
SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > ROM
  .rodata : {
    *(.rodata .rodata.* .srodata .srodata.*)
    *(.init_array .init_array.* .fini_array .fini_array.* .eh_frame .eh_frame_hdr)
  } > DROM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .tdata : { __tls_base = .; *(.tdata .tdata.*) } > RAM
  .tbss : { *(.tbss .tbss.*) } > RAM
  .bss : {
    . = . + SIZEOF(.tbss);  /* (.tbss takes no address space of its own) */
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(16);
    __heap_start = .;
  } > RAM
  __heap_end = 0x7000;  /* 4 KB of stack above the heap */
  _end = __heap_start;
}
