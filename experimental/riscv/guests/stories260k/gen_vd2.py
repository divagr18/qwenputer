"""vd2.S: vk_<np>_<init|acc>_<raw|scaled>(int32_t *out, int wb, int rows, const int32_t *S, int F).

Rows of np words (8 np int4 weights) stored back to back from WROM word wb; 4 rows per WB (the
VDOTs walk straight through them). out[r] (= 0 or its old value) += x . row r; scaled: then
out[r] = mulh(out[r] * S[r], F). The 2 np activation words are loaded from the pointer vk_x."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REGS = [f"s{i}" for i in range(12)] + ["t3", "t4", "t5", "t6"]
ACC = ["a5", "a6", "a7", "t0"]
out = ["  .text"]
for np_ in (8, 6):
    for init in (True, False):
        for scaled in (True, False):
            f = f"vk_{np_}_{'init' if init else 'acc'}_{'scaled' if scaled else 'raw'}"
            fn = [f"  .globl {f}", f"{f}:", "  addi sp, sp, -48"]
            fn += [f"  sw s{i}, {4 * i}(sp)" for i in range(12)]
            fn += ["  lui t2, %hi(vk_x)", "  lw t2, %lo(vk_x)(t2)"]
            fn += [f"  lw {REGS[i]}, {4 * i}(t2)" for i in range(2 * np_)]
            fn += ["1:"]
            fn += [f"  li {a}, 0" if init else f"  lw {a}, {4 * r}(a0)" for r, a in enumerate(ACC)]
            fn += ["  .insn r 0x0B, 6, 0, zero, a1, zero"]  # wb a1
            for a in ACC:
                fn += [f"  .insn r 0x0B, 5, 0, {a}, {REGS[2 * j]}, {REGS[2 * j + 1]}" for j in range(np_)]
            for r, a in enumerate(ACC):
                if scaled:
                    fn += [f"  lw t1, {4 * r}(a3)", f"  mul {a}, {a}, t1", f"  mulh {a}, {a}, a4"]
                fn += [f"  sw {a}, {4 * r}(a0)"]
            fn += ["  addi a0, a0, 16", f"  addi a1, a1, {4 * np_}"]
            fn += ["  addi a3, a3, 16"] if scaled else []
            fn += ["  addi a2, a2, -4", "  bnez a2, 1b"]
            fn += [f"  lw s{i}, {4 * i}(sp)" for i in range(12)]
            fn += ["  addi sp, sp, 48", "  ret"]
            out += fn
open(f"{HERE}/vd2.S", "w").write("\n".join(out) + "\n")
