# Parallel Scientific Computing

A reproducible computational performance project using **Python, NumPy, and
multiprocessing** to simulate heat diffusion and compare sequential and parallel
execution. No external datasets or services are required.

## Quick start

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python -m parallel_science --size 128 --steps 200 --trials 8 --workers 1 2 --repeats 3
```

The same command is available as `parallel-science` after installation.
Windows users can activate with `.venv\Scripts\activate`.
Dependencies are pinned in `requirements.txt` for repeatable development installs.

## Scientific model

The simulation solves the 2D heat equation on a square, unit-spaced grid with
fixed zero-temperature boundaries. Each step applies the five-point stencil:

```text
u_next[i,j] = u[i,j] + alpha * (
    u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]
)
```

Here alpha combines diffusivity, timestep, and grid spacing. The explicit scheme
requires 0 < alpha <= 0.25; the default is 0.2. Initial interior temperatures are
uniform random values in [0, 1). NumPy vectorizes each update and two arrays
alternate between steps. This is an educational finite-difference model, not a
calibrated physical prediction.

Parallelism is across independent ensemble trials, not spatial subdomains.
Trial i always uses seed + i, so scheduling and worker count do not change the
work. Each worker returns mean temperature and sum of squared temperatures;
full arrays stay in the worker. The sequential baseline uses the identical
numerical kernel. The process pool uses spawn for cross-platform behavior.

## Benchmarking and profiling

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m parallel_science --size 256 --steps 500 --trials 12 \
    --workers 1 2 4 --repeats 5 --output results/scaling.json \
    --profile results/sequential.prof
python -m pstats results/sequential.prof
```

In pstats, use `sort cumulative` then `stats 20` to inspect hotspots.
The profile is a separate sequential run, so instrumentation does not contaminate
benchmark timings. It does not profile subprocesses.

The CLI checks every timed result against an untimed sequential reference
(rtol and atol 1e-12), prints a table, and saves JSON containing all timing samples,
median runtime, speedup, efficiency, problem parameters, and runtime metadata.
Worker count 1 is always included as the baseline. Requested worker counts are
capped to the number of trials; efficiency uses the active worker count.

- Speedup = sequential median / parallel median.
- Efficiency = speedup / active workers.
- Timing covers pool creation, simulation, IPC, and shutdown; reference checks,
  JSON output, and profiling are excluded. There is no warmup.
- Memory scales approximately with active workers times grid area; each worker
  holds two persistent grids plus temporary stencil arrays.

Small workloads often run faster sequentially because process startup dominates.
Larger workloads may benefit, but CPU quotas, memory bandwidth, other activity,
and oversubscription limit scaling. Repeat runs on an otherwise idle machine;
do not treat one timing as a general performance claim. Thread limits above
reduce accidental library oversubscription, though this stencil uses no BLAS.

## Layout and validation

- `parallel_science/simulation.py`: numerical kernel and ensemble execution.
- `parallel_science/cli.py`: correctness-checked timing, JSON export, profiling.
- `tests/test_simulation.py`: stencil values, boundary conditions, dissipation,
  seed determinism, invalid parameters, input preservation, and process parity.
- `.github/workflows/tests.yml`: Python 3.10/3.12 tests and CLI smoke run.

Generated benchmark and profile files live in ignored `results/`.
Run tests before interpreting performance. There is no web server to start.
