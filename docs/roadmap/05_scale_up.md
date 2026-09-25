# 05 · Scale-up: GPU, real text, 100M → 1B → 4B

## Built
- `recursionx/data/text.py`:
  - `ByteTokenizer`, with ids < 32 reserved to match the special/task tokens
    of the skill suite.
  - `load_tokenizer("hf:<name>")`, which needs `transformers`.  **This path
    is untested here.**
  - `build_bin`: txt/md/jsonl → flat uint16/uint32 token file + `.val` + `.meta.json`.
  - `TokenBin`: memory-mapped random windows.
- `recursionx/presets.py`: `tiny / small / base / 1b / 4b`.  `describe(name)`
  prints exact counts via meta-device instantiation.  With a 50,304-token
  vocabulary:

| preset | unique params | active / pass | Engram table (RAM/disk) | effective depth @R=4 |
|---|---|---|---|---|
| small | 62.8M | 36.2M | 17.0M | 14 blocks |
| base | 134.1M | 68.0M | 67.4M | 16 blocks |
| 1b | 1.17B | 287M | 538M | 28 blocks |
| 4b | 3.72B | 606M | 2.15B | 28 blocks |

- `experiments/pretrain_text.py`:
  - `prepare` / `train` subcommands, device auto-select, bf16/fp16 autocast
    with GradScaler, gradient accumulation, cosine LR + warmup, weight-decay
    groups (none on the Engram table), validation loss, samples,
    checkpoint/resume, `--compile`.
  - Smoke-tested on CPU only (bf16 autocast): loss finite, checkpoint written.
- The liquid scan and the Titans memory updates run in fp32 even under
  autocast, because they accumulate over the sequence.
- Every `collate`/eval path moves tensors to the model's device, so the
  lifecycle code is device-agnostic.  It has not been run on CUDA yet.

## Placeholders / tasks (in priority order)
1. **First GPU run.**  `small` on ~1B tokens of a clean corpus (FineWeb-Edu
   sample or similar).  Report tokens/s and val loss.  Run the same config
   with `n_loops=1` and with `use_engram=False` / `use_neural_memory=False`
   as ablations, now on real text, where Engram and Titans *should* help.
2. **Fused MoE.**  `FluidMoE.forward` loops over experts in Python, which is
   fine on CPU and slow on GPU.  Replace it with sorted-token grouped GEMM
   (`torch._grouped_mm` if available, MegaBlocks or ScatterMoE).  Keep the
   loop as the reference implementation and add an equivalence test.
3. **Liquid scan kernel.**  The chunked scan is O(T·c) memory per chunk.
   For long sequences, write a Triton kernel or use an associative scan
   (`torch._higher_order_ops.associative_scan`).  Add an equivalence test
   against `liquid_scan`.
4. **KV cache / incremental decoding.**  `generate` recomputes the prefix.
   State needed per layer: the KV cache (attention), the last `conv_kernel-1`
   inputs plus scan state `h` (liquid), memory `(M, S)` plus the
   partial-chunk buffer (Titans), the last `max(orders)-1` tokens (Engram),
   and **per loop iteration** for the core.  So the core cache is
   `n_loops ×` larger.  Verify that cached and uncached logits match.
5. **Deep Titans memory.**  Replace the linear memory with a 2-layer MLP
   updated via `torch.func.grad` / `vmap` per sequence.  Also try
   per-token (not per-chunk) momentum.
6. **Engram at scale.**  Tables of 10⁸+ rows on disk: prefetch rows for the
   next batch on a background thread, keep an LRU of hot rows on the GPU,
   and use sparse Adam (or Adagrad) for disk rows instead of plain SGD.
   Tokenizer-normalised n-grams (lower-case, NFKC), as in the Engram paper.
7. **Distributed.**  FSDP2 (`fully_shard`) for `1b`/`4b`.  Put experts on an
   expert-parallel group once the fused kernel exists.  Checkpoint with
   `torch.distributed.checkpoint`.
8. **Lifecycle at scale.**  The skill suite is replaced by instruction data.
   Base skills become the pre-training/SFT distribution, and new skills are
   datasets arriving over time (new domains, tools, languages).  Build the
   evaluation harness: per-skill probes, plus a general-capability suite
   (perplexity on held-out web text, a few standard benchmarks) as the
   "base retention" metric.

## Acceptance criteria
- A `small` GPU run with a loss curve and ablations in `results/text/`.
- Fused MoE and fast scan with equivalence tests.  Throughput numbers before
  and after.
- KV-cached `generate` matching uncached logits (`atol 1e-3` in bf16), with
  a test.
