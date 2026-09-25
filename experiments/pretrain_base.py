"""Pre-train the small Recursion-X base model on the base skill suite."""
import argparse

import torch

from common import pretrain

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    pretrain(steps=args.steps, seed=args.seed, out=args.out)
