#!/bin/bash
# Everything that needs memory or the network, one job at a time (they OOM together on this machine).
ROOT=/c/Users/kesha/AppData/Local/Temp/claude/D--weights/d1310d09-a8d6-4b26-affc-4f555296c2cb/scratchpad
SP="C:/Users/kesha/AppData/Local/Temp/claude/8450bc6024565175/d1310d09-a8d6-4b26-affc-4f555296c2cb/scratchpad/size"
TK=$(ls -d /c/Users/kesha/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf8
D=$ROOT/qwenputer-size/experimental/riscv/fx/size
echo "chain start $(date)"
(cd $ROOT/qwenputer-exec/experimental/riscv && PYTHONPATH=. python $D/verify_sparse.py "$TK" "$SP" q4p2 > "$SP/verify_q4p2.log" 2>&1); echo "verify q4p2 rc=$? $(date)"
(cd $ROOT/qwenputer-exec/experimental/riscv && PYTHONPATH=. python $D/analyze.py "$TK" legacy "$SP/an_q4p2.pkl" > "$SP/an_q4p2.log" 2>&1); echo "analyze q4p2 rc=$?"
(cd $ROOT/qwenputer-size/experimental/riscv && PYTHONPATH=. python $D/analyze.py "$TK" legacy "$SP/an_fxint.pkl" > "$SP/an_fxint.log" 2>&1); echo "analyze fxint rc=$?"
(cd $ROOT/qwenputer-mem/experimental/riscv && PYTHONPATH=. python $D/analyze.py "$TK" fx "$SP/an_fx.pkl" > "$SP/an_fx.log" 2>&1); echo "analyze fx rc=$?"
(cd $ROOT/qwenputer-mem/experimental/riscv && PYTHONPATH=. python $D/verify_sparse.py "$TK" "$SP" fx > "$SP/verify_fx.log" 2>&1); echo "store fx rc=$?"
python $D/sample.py "$SP" --base-only > "$SP/sample_base.log" 2>&1; echo "sample rc=$? $(date)"
for i in 1 2 3 4 5 6; do
  (cd $ROOT/qwenputer-size/experimental/riscv && "$SP/venv/Scripts/python.exe" -W ignore fx/size/qstream.py "$SP" "$TK" >> "$SP/qstream.log" 2>&1) && break
  echo "qstream stopped, try $i $(date)"; sleep 60
done
echo "chain end $(date)"
