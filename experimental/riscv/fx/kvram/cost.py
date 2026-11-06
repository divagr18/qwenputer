"""Cost model: KV-cache RAM vs DeltaNet word slots for fx (all numbers here are estimates, E,
computed from the shapes below; the fx counts come from the measured trace, M).

  python cost.py OUT_JSON
"""
from __future__ import annotations

import json
import sys

# Qwen3.5-4B full-attention layer (config.json)
N_KV, N_Q, HD = 4, 16, 256
BF16 = 2
# NVIDIA L4
L4_BYTES = 24e9
L4_BW = 300e9
L4_BF16_FLOPS = 121e12  # dense tensor-core peak
MODEL_BYTES = 18.7e9  # rv-q4p2 (11 chip layers + Qwen3.5-4B) in bf16 (docs/rv32.md)
LAYER_BYTES = 0.3e9  # one extra layer at the chip's hidden size (brief)
# fx session (measured on the trace, gen_trace.py)
FX = {"stores": 5_402_052, "loads": 8_740_595, "instr": 38_024_738, "distinct_words": 165_315,
      "live_nonzero_words_end": 93_278, "written_kb": 630, "live_kb": 351}
TOK_PER_INSTR = (1.0, 1.3, 1.55)  # P0: turn 1 is 33-50M trace tokens for 32.5M instructions (E)
GDN_WORDS_PER_LAYER = 4096  # 16 KB of word slots per DeltaNet layer (rv32.md)
GDN_STATE_BYTES = 32 * 128 * 128 * 4  # fp32 recurrent state of one Qwen3.5 DeltaNet layer


def row(entries_stored, bytes_per_entry, heads, entries_read_per_head):
    """KV memory and per-token attention cost: flops = heads * 4 * HD * entries (QK and PV)."""
    kv = entries_stored * bytes_per_entry
    rd = entries_read_per_head * bytes_per_entry  # GQA: the 4 q heads of a kv head share one read
    fl = heads * 4 * HD * entries_read_per_head
    return {"kv_bytes": kv, "bytes_read_per_token": rd, "flops_per_token": fl,
            "ms_bandwidth": 1e3 * rd / L4_BW, "ms_compute": 1e3 * fl / L4_BF16_FLOPS,
            "fits_L4_beside_model": kv <= L4_BYTES - MODEL_BYTES}


def main():
    out = {"labels": "E (estimates from shapes); fx counts M", "fx": FX}
    hf_entry = N_KV * HD * 2 * BF16  # K and V of every kv head, every position
    out["hf_bytes_per_position_per_layer"] = hf_entry
    pos = {k: FX["instr"] * k for k in TOK_PER_INSTR}
    out["fx_positions_estimate"] = pos
    out["hf_bytes_per_stored_word"] = {str(k): hf_entry * p / FX["stores"] for k, p in pos.items()}
    out["hf_session_kv_bytes"] = {str(k): hf_entry * p for k, p in pos.items()}
    out["hf_positions_that_fit_L4"] = int((L4_BYTES - MODEL_BYTES) / hf_entry)
    rows = {}
    for N in (10 ** 5, 10 ** 6, 10 ** 7):
        rows[f"1e{len(str(N)) - 1}"] = {
            # HF path: N positions in the cache (4 kv heads each); every token runs all 16 q heads over them
            "hf_dynamic_cache_N_positions": row(N, N_KV * 2 * HD * BF16, N_Q, N),
            # executor: N store entries, one kv head each (256-dim K and V), 4 banks: a load reads N/4
            "store_only_N_stores": row(N, 2 * HD * BF16, 1, N // 4),
        }
    out["per_load_cost"] = rows
    live = FX["distinct_words"]
    out["dedup_store_only"] = row(live, 2 * HD * BF16, 1, live // 4) | {"entries": live}
    out["hash_indexed"] = {"bytes_per_word": 8, "session_bytes": live * 8, "cost_per_load": "O(1), one gather"}
    out["deltanet_word_slots"] = {
        "layers_for_written": -(-FX["distinct_words"] // GDN_WORDS_PER_LAYER),
        "layers_for_live": -(-FX["live_nonzero_words_end"] // GDN_WORDS_PER_LAYER),
        "brief_range_layers": [16, 40],
        "checkpoint_gb": [-(-FX["live_nonzero_words_end"] // GDN_WORDS_PER_LAYER) * LAYER_BYTES / 1e9,
                          -(-FX["distinct_words"] // GDN_WORDS_PER_LAYER) * LAYER_BYTES / 1e9],
        "runtime_state_mb": [-(-FX["live_nonzero_words_end"] // GDN_WORDS_PER_LAYER) * GDN_STATE_BYTES / 1e6,
                             -(-FX["distinct_words"] // GDN_WORDS_PER_LAYER) * GDN_STATE_BYTES / 1e6],
        "note": "slots are direct-mapped by address bits, so the count assumes a compacted address map",
    }
    out["kv_layer_checkpoint_gb"] = LAYER_BYTES / 1e9
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
