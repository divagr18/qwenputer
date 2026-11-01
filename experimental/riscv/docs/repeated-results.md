# Reusing calculator results

`--acknowledge-repeats` optionally interrupts repeated successful calculations:

```sh
python -m chipformer.rv.chat qwenputer-rv --slow \
  --firmware tests/data/rv/calc-bridge.elf \
  --ask "What is 71898 * 59756?" --acknowledge-repeats
```

After the second successful return for the same expression, the controller selects:

> This calculation is already complete. I will use the returned result.

Normal generation then continues. The controller reads the actual circuit-owned
response and selects ordinary text tokens through the optional `on_step` callback.
The calculator still performs the arithmetic and emits its result. Different
expressions, error responses and model-written imitations of a tool response do
not trigger an acknowledgement. Whitespace is ignored when comparing expressions.
The flag is off by default.

For an existing `RvChat` application:

```python
from chipformer.rv.repeat_results import RepeatedResultAcknowledgement

control = RepeatedResultAcknowledgement.for_chat(chat)
result = chat.generate(prompt_ids, parked_machine, on_step=control)
```

Create a new controller for each generation. It stops intervening after 128
completed calls by default; `max_calls` can be set explicitly. Oversized or
unparsed calls receive no acknowledgement.

On the original `arith(40, 0)` bank, the standalone controller improved correct
answers from 36/40 to 40/40 and reduced completed redundant calls from 188 to 7.
All four cases that exhausted the original text budget answered correctly after
two calls with the controller. No correct answer regressed; all 33 inactive
pairs had identical complete traces.

The paired run used the original prompts, 4096 text / 200000 chip-token budgets,
no call cutoff, greedy decoding and a fresh cache per question. Runtime: Triton
circuit executor, eager/reference backbone, PyTorch 2.13.0, Transformers 5.17.0
and Triton 3.7.1; checkpoint `2d9068508f2b680badb7abcded96fdd107e9ba5d`.
All 80 traces passed independent firmware and controller replay. Eight native
executor compatibility cases also matched their controlled HF traces. The fast
backend and newer upstream dependency versions were not measured here.

From `experimental/riscv`, the focused tests require only Python:

```sh
python -m unittest discover -s tests -p test_rv_repeat_results.py
python -m unittest discover -s tests -p test_rv_next_token_control.py
```
