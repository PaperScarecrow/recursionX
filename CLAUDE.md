# Recursion-X: notes for Claude Code

Research prototype of a continually learning transformer.  Read `README.md`
for results and `docs/DESIGN.md` for the architecture.  Work items and QA
instructions live in `docs/roadmap/`; start with `docs/roadmap/README.md`.

## Layout
- `recursionx/`: the library.  `model.py` (backbone), `modules/` (liquid,
  attention, moe, neural_memory, engram, lora), `lifecycle/` (wake, gate,
  sleep, brain, skills), `research/` (sources, verifiers, loop), `skills/`
  (hyperlora), `data/` (synthetic tasks, text pipeline), `presets.py`,
  `train.py`.
- `experiments/`: runnable scripts.  They import `common.py` and write to
  `runs/`, which is gitignored.  Curated outputs are copied to `results/`.
- `tests/`: pytest.  The full suite takes about 2 minutes on 4 CPU cores.

## Commands
```bash
pip install -e .[dev]
pytest -q                                        # must stay green
cd experiments
python pretrain_base.py --steps 3000             # base model for skill benchmarks -> runs/base.pt
python continual.py --methods rx --seed 0        # one continual run (~90 CPU-min)
python run_matrix.py --methods rx,rx_audit,finetune_replay --seeds 0,1,2 --parallel 2
python report.py --plot                          # aggregate runs/continual/*.json
python pretrain_text.py prepare --data <dir> --out ../runs/text/corpus.bin
python pretrain_text.py train --bin ../runs/text/corpus.bin --preset small --device cuda
python hyperlora.py                              # HyperLoRA meta-training + warm-start test
```

## Conventions
- Every weight that can learn a skill is an `AdaptableLinear`.  New layers
  should use it instead of `nn.Linear`, unless they must never be adapted
  (for example routers).
- The model is causal.  `tests/test_modules.py::test_model_is_causal` must
  pass for every new mixer or memory option.
- Report results honestly, including negative ones.  A README claim needs a
  JSON file under `results/` behind it.
- Keep runs reproducible: seeds via `common.seed_all`, and evaluation sets
  hashed with `zlib.crc32`, never Python `hash()`.
- CPU runs use `--threads 2` per process with two processes in parallel.
  More processes oversubscribe the 4 cores and slow everything down.
