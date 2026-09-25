"""Run a (method × seed) matrix of continual-learning experiments in parallel
worker processes, then aggregate with report.py.

  python run_matrix.py --methods rx,rx_audit,finetune_replay --seeds 0,1,2 \
      --stream long --parallel 2 --threads 2
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--stream", default="default")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    out = os.path.join(HERE, "..", "runs", "continual" if args.stream == "default" else f"continual_{args.stream}")
    logs = os.path.join(HERE, "..", "runs", "logs")
    os.makedirs(logs, exist_ok=True)
    jobs = []
    for m, s in itertools.product(args.methods.split(","), args.seeds.split(",")):
        if args.skip_existing and os.path.exists(os.path.join(out, f"{m}_s{s}.json")):
            continue
        jobs.append((m, s))
    running = []
    while jobs or running:
        while jobs and len(running) < args.parallel:
            m, s = jobs.pop(0)
            log = open(os.path.join(logs, f"{args.stream}_{m}_s{s}.log"), "w")
            cmd = [sys.executable, "continual.py", "--methods", m, "--seed", s, "--stream", args.stream,
                   "--threads", str(args.threads)]
            running.append(((m, s), subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)))
            print(f"started {m} seed {s}", flush=True)
        time.sleep(5)
        for job in list(running):
            (m, s), p = job
            if p.poll() is not None:
                print(f"finished {m} seed {s} (exit {p.returncode})", flush=True)
                running.remove(job)
    subprocess.run([sys.executable, "report.py", "--dir", out, "--plot"], cwd=HERE)
