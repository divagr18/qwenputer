ENTRY(_start)
SECTIONS
{
  . = 0x00010000;
  .text : { *(.text.start) *(.text .text.*) }
  . = ALIGN(16);
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) }
  . = ALIGN(16);
  .data : { *(.data .data.* .sdata .sdata.*) }
  . = ALIGN(16);
  .bss : { *(.bss .bss.* .sbss .sbss.* COMMON) }
  . = ALIGN(16);
  _end = .;
  ASSERT(_end < 0x01C00000, "image overlaps the stack")
  /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.note*) *(.eh_frame*) }
}
