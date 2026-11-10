"""Benchmark and profile the same deterministic ensemble."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from .simulation import Simulation, run


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=positive, default=128)
    parser.add_argument("--steps", type=positive, default=200)
    parser.add_argument("--trials", type=positive, default=8)
    parser.add_argument("--workers", type=positive, nargs="+", default=[1, 2])
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--profile", type=Path, help="cProfile output for a separate sequential run")
    args = parser.parse_args()
    try:
        config = Simulation(args.size, args.steps)
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
    except ValueError as exc:
        parser.error(str(exc))
    reference = run(config, args.trials, args.seed)  # untimed correctness reference
    rows = []
    for workers in sorted(set([1] + args.workers)):
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            observed = run(config, args.trials, args.seed, workers)
            samples.append(time.perf_counter() - start)
            np.testing.assert_allclose(observed, reference, rtol=1e-12, atol=1e-12)
        median = statistics.median(samples)
        baseline = rows[0]["median_seconds"] if rows else median
        active = min(workers, args.trials)
        rows.append(dict(workers=workers, active_workers=active, seconds=samples,
                         median_seconds=median, speedup=baseline / median,
                         efficiency=baseline / median / active))
    report = dict(
        simulation=vars(config), trials=args.trials, seed=args.seed,
        repeats=args.repeats, python=platform.python_version(), numpy=np.__version__,
        platform=platform.platform(), logical_cpus=os.cpu_count(),
        thread_environment={key: os.environ.get(key) for key in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        timing_scope="End-to-end ensemble, including pool startup and IPC; no warmup",
        correctness="All observables match sequential reference within 1e-12",
        results=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("workers  median (s)  speedup  efficiency")
    for row in rows:
        print(f'{row["workers"]:7d}  {row["median_seconds"]:10.4f}  '
              f'{row["speedup"]:7.2f}  {row["efficiency"]:10.1%}')
    print(f"Results: {args.output}")
    if args.profile:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.runcall(run, config, args.trials, args.seed, 1)
        profiler.dump_stats(str(args.profile))
        print(f"Sequential profile: {args.profile}")


if __name__ == "__main__":
    main()
