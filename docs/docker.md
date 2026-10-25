# Docker: run selected training and evaluation jobs

One image contains the dependencies for the model families in this repository.
You select a Python script and its arguments for each container invocation.
Building or starting the image does not launch a training suite. Sequential
commands use one container at a time; separate terminals can run independent
jobs concurrently.

The environment uses Python 3.12.13, PyTorch 2.14.1 with CUDA 12.6, BayesFlow
2.0.14 and Keras 3.15.1. Python 3.12 supports the BayesFlow dependency; the
existing Windows Python 3.14 virtual environment is not copied into the image.
The other direct package pins match the current local environment. Exact
resolved Python dependencies are recorded in `requirements-docker.lock`.
The native `requirements.txt` is unchanged.
The image also includes the C/C++ compiler tools needed by Triton's runtime
GPU kernel helpers.

## Build once

On Windows, start Docker Desktop with Linux containers and its WSL 2 backend.
GPU jobs require a compatible NVIDIA Windows driver. See the official
[Docker GPU prerequisites](https://docs.docker.com/desktop/features/gpu/).

Open PowerShell in the repository root:

```powershell
docker build -t inverse-benchmark .
```

The first build downloads PyTorch and the CUDA user-space libraries and can be
large. Subsequent builds reuse the dependency layer when the lock file is
unchanged. The image includes executable source, tests and the two fixed test
condition files; local environments, checkpoints, caches and results are
excluded by `.dockerignore`.

Check the installed dependencies and GPU access:

```powershell
docker run --rm inverse-benchmark python -m pip check
docker run --rm --gpus all inverse-benchmark python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
```

## Select a training job

These commands mount the current project directory at `/app`. Consequently,
the scripts use your current source code and their output files are written
directly into your local project. Dependencies come from the image's Linux
Python, not from the Windows `.venv`.

For example, train only FIF's paper architecture on ballistics:

```powershell
docker run --rm --gpus all `
  --mount "type=bind,source=$($PWD.Path),target=/app" `
  inverse-benchmark `
  python experiments/train_fif.py --problem ballistics --arch paper --run bal_fif_p3_docker_s0 --device cuda
```

Or train only FMPE on kinematics:

```powershell
docker run --rm --gpus all `
  --mount "type=bind,source=$($PWD.Path),target=/app" `
  inverse-benchmark `
  python experiments/train_fmpe.py --problem kinematics --run-name kin_fmpe_docker_s0 --device cuda
```

Run either command, or both sequentially, according to the jobs you want.
Choose a new run name for a new training experiment. Script arguments remain
unchanged: FIF uses `--run`; FMPE uses `--run-name`.

For FIF on kinematics, use `--problem kinematics` and a run name such as
`kin_fif_p3_docker_s0`. For the original FIF MLP, use `--arch mlp`.

## Select an evaluation job

Evaluate a saved FIF run independently of training:

```powershell
docker run --rm --gpus all `
  --mount "type=bind,source=$($PWD.Path),target=/app" `
  inverse-benchmark `
  python experiments/evaluate_fif.py bal_fif_p3_docker_s0 --device cuda
```

Evaluate a saved FMPE run with its chosen solver:

```powershell
docker run --rm --gpus all `
  --mount "type=bind,source=$($PWD.Path),target=/app" `
  inverse-benchmark `
  python experiments/evaluate_fmpe.py kin_fmpe_docker_s0 --solver rk4:8 --name kin_fmpe_docker_s0_eval1 --device cuda
```

You can evaluate existing runs by substituting their names, for example
`kin_fif_s0` or `kin_fmpe_s0`. Their checkpoint and configuration files must be
present in the mounted project. For repeated FMPE evaluations, choose a new
`--name` to keep previous results; FIF automatically creates a fresh evaluation
directory.

To inspect any other model's available arguments, run its script with
`--help` using the same image:

```powershell
docker run --rm inverse-benchmark python experiments/train_inn.py --help
docker run --rm inverse-benchmark python experiments/bayesflow_npse.py --help
```

BayesFlow scripts combine training and optional evaluation. For example,
`bayesflow_npse.py --problem kinematics --run-name kin_bf_npse_docker_s0 --sampler`
trains only, because the empty `--sampler` list disables evaluation. Add
`--evaluate-only --sampler euler:32` to evaluate that saved run instead.

## Persistent files

With the project mounted at `/app`, these locations are on your computer:

| Files | Host project path | Container path |
| --- | --- | --- |
| Training checkpoints, configuration and logs | `experiments/runs/<run>/` | `/app/experiments/runs/<run>/` |
| FIF evaluation results | `experiments/runs/<run>/evaluation/` (then `evaluation_2/`, etc.) | `/app/experiments/runs/<run>/evaluation/` |
| Shared evaluation results, including FMPE | `metrics/results/` | `/app/metrics/results/` |
| Ground-truth posterior cache | `metrics/cache/` | `/app/metrics/cache/` |
| Plots and reports, when generated by a script | The script's `docs/` or `output/` destination | `/app/docs/` or `/app/output/` |

`--rm` removes the finished container, while files written through the bind
mount remain on the host. Always include the mount when saving jobs or reading
your existing runs. Without it, the image's own source code is used and files
created inside the disposable container are removed with the container.

## CPU jobs

Use the same image, omit `--gpus all`, and select `--device cpu`:

```powershell
docker run --rm `
  --mount "type=bind,source=$($PWD.Path),target=/app" `
  inverse-benchmark `
  python experiments/evaluate_fif.py kin_fif_s0 --device cpu
```

## Update the dependency lock

Edit the direct pins in `requirements-docker.in`, then regenerate the lock
with [uv](https://docs.astral.sh/uv/pip/compile/). From a Linux Python 3.12
environment, or from Windows with a local Python 3.12 available:

```powershell
uv pip compile requirements-docker.in --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --index-strategy unsafe-best-match --emit-index-url --output-file requirements-docker.lock
```

The additional index supplies the explicitly pinned `+cu126` PyTorch wheel.
The remaining Python packages are resolved to exact versions. Rebuild the
image after changing the lock. Containerization fixes the software environment;
identical GPU results also depend on the hardware, driver and random seeds.

## Run the existing tests

The image includes the test suite. Run BayesFlow's tests in a separate process,
because importing BayesFlow changes PyTorch's process-wide autograd state:

```powershell
docker run --rm -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 inverse-benchmark python -m pytest -q --ignore=tests/test_npse.py
docker run --rm -e KERAS_TORCH_DEVICE=cpu inverse-benchmark python -m pytest -q tests/test_npse.py
```

Saved-run tests skip when their checkpoints are not present; checkpoints are
deliberately excluded from the image. The regular model and metric tests run
against the code packaged in the image.

Validation on 2026-10-25 with a GTX 1650 Ti: real CUDA computation and autograd
passed. Four small FIF jobs (MLP/paper on kinematics/ballistics) completed GPU
training, checkpoint reload and pilot evaluation. Their checkpoints and metric
summaries remained on the host after the container exited.

The two test commands above produced 142 passes, 2 skips and 1 failure in total.
The remaining failure is
`tests/test_fff.py::test_torus_learns_distribution_across_the_cut`, a learning
convergence check. That test also fails in the existing Windows `.venv` outside
Docker.

For Linux hosts, the same Docker commands work with a Linux-style bind mount,
such as `--mount "type=bind,source=$(pwd),target=/app"`. For large workloads on
Windows, a working copy in WSL's Linux filesystem can reduce mount overhead;
see [Docker's WSL filesystem guidance](https://docs.docker.com/desktop/features/wsl/best-practices/).
