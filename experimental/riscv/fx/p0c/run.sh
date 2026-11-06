#!/bin/bash
# P0c runner: native fx v0.0.16 + base Qwen3.5-4B (vLLM) on 5 tiny-repo tasks.
# usage: run.sh COND REPS   (COND = full: all tools, --full-access; noshell: bash denied, auto mode)
set -u
P=/opt/dlami/nvme/p0c
FX=$P/fxbin/fx
COND=$1; REPS=${2:-3}
OUT=$P/results/$COND; mkdir -p $OUT
mkdir -p ~/.fx
if [ "$COND" = noshell ]; then
  PERM=',"permission": {"bash": {"*": "deny"}}'; MODE=""
else
  PERM=''; MODE="--full-access"
fi
cat > ~/.fx/settings.json <<EOF
{"provider": "local",
 "providers": {"local": {"protocol": "openai-chat-completions", "base_url": "http://127.0.0.1:8000/v1",
   "auth": {"type": "none"},
   "model_metadata": {"qwen3.5-4b": {"context_window": 65536, "max_output_tokens": 8192, "supports_tool_use": true}}}},
 "models": {"local": "qwen3.5-4b"}$PERM}
EOF
for r in $(seq 1 $REPS); do
  for t in t1_bug t2_rename t3_feature t4_config t5_parse; do
    W=$P/work/$COND-$r; rm -rf $W; mkdir -p $W; python3 $P/tasks.py make $W
    prompt=$(python3 -c "import json;print(json.load(open('$W/tasks.json'))['$t'])")
    s=$(date +%s)
    (cd $W/$t && timeout 900 $FX ask $MODE --json --no-save -- "$prompt" > $OUT/$t-$r.json 2> $OUT/$t-$r.err < /dev/null)
    rc=$?
    dt=$(( $(date +%s) - s ))
    res=$(python3 $P/tasks.py check $W $t 2>&1); ok=$?
    (cd $W/$t && git diff --stat > $OUT/$t-$r.diffstat; git status --short >> $OUT/$t-$r.diffstat; git diff > $OUT/$t-$r.diff)
    echo "{\"cond\":\"$COND\",\"task\":\"$t\",\"rep\":$r,\"fx_rc\":$rc,\"seconds\":$dt,\"solved\":$([ $ok = 0 ] && echo true || echo false),\"check\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$res")}" | tee -a $P/results/summary.jsonl
  done
done
