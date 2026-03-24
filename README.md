# Ordering-UCT for Continuous Causal Discovery

This repository is the trimmed, continuous-variable-only codebase built around the causal discovery code from [_Tree search in DAG space with model-based reinforcement learning for causal discovery_](https://doi.org/10.1098/rspa.2024.0450). The active code now focuses on:

- edge-based CD-UCT
- ordering-based CD-Ordering-UCT
- a random-ordering baseline
- the datasets, experiment outputs, and plotting scripts used to compare them

The main implementation lives in `code/causal-discovery`. The repository also ships with datasets in `datasets/` and checked-in experiment artifacts in `experiments/causal-discovery/`.

## What Is In This Repo

The repository is no longer the full paper companion tree. The current codebase is organized around the continuous-variable workflow only.

```text
code/causal-discovery/
  cdrl/
    run_causal_discovery.py      # single dataset entry point
    setup_experiments.py         # create hyperopt/eval task files
    tasks.py                     # execute one serialized task
    agent/
      mcts/                      # edge-based UCT agents
      ordering/                  # ordering-based UCT agents
      baseline/                  # random-ordering baseline
  test/                          # smoke tests
  scripts/recipes/               # batch experiment helpers
  docker/                        # legacy container setup
  environment.yml                # legacy conda environment spec

datasets/                        # source datasets checked into the repo
experiments/causal-discovery/    # copied datasets, metrics, figures, task outputs
```

## Supported Algorithms

`cdrl/run_causal_discovery.py` exposes three algorithms:

- `uct`: edge-based Monte Carlo tree search over graph edits
- `ordering_uct`: ordering-based Monte Carlo tree search over variable orderings
- `random_ordering`: random ordering baseline

It also supports:

- `both`: run `uct` and `ordering_uct` together
- `all`: run `uct`, `ordering_uct`, and `random_ordering` together

In the task-based experiment pipeline, the full-depth edge-based agent is written to disk as `uctfull` in metrics filenames. That is the same method that the single-run CLI refers to as `uct`.

## Datasets Wired Into The Code

The current instance generators recognize the following continuous-variable datasets:

- `datasets/sachs/`: Sachs benchmark, Gaussian process regression (`GPR`)
- `datasets/syntren/1` ... `datasets/syntren/10`: SynTReN graphs, `GPR`
- `datasets/synthetic/*nodeslr/gauss_same_noise/1`: linear synthetic graphs, `LR`
- `datasets/synthetic/50nodesqr/gauss_same_noise/1`: quadratic synthetic graph, `QR`
- `datasets/gpgen/`: Gaussian-process-generated families used by the task pipeline

For containerized runs, the code expects these files under `/experiment_data/datasets/...`. This repository already includes a ready-to-mount experiment root at `experiments/causal-discovery/`, including a `datasets/` copy.

## Setup

### Recommended workflow

The intended workflow is still Docker-first because the historical environment depends on:

- Python 3.6
- TensorFlow 1.13
- R packages used by CAM
- Graphviz / PyGraphviz

Set the two paths used by the scripts:

```bash
export CD_SOURCE_DIR="$PWD/code/causal-discovery"
export CD_EXPERIMENT_DATA_DIR="$PWD/experiments/causal-discovery"
```

On Apple Silicon, the original environment was built under `linux/amd64`, so you may also need:

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

### Build and start the manager container

```bash
bash "$CD_SOURCE_DIR/scripts/update_container.sh"
bash "$CD_SOURCE_DIR/scripts/manage_container.sh" up
```

Stop it with:

```bash
bash "$CD_SOURCE_DIR/scripts/manage_container.sh" stop
```

### Important note about the legacy environment

The active causal discovery code no longer uses the old RL-BIC path, but parts of the historical Docker scaffolding still reference it. Treat the Dockerfiles and `environment.yml` as the original runtime baseline for this trimmed repository, not as a freshly curated minimal environment.

## Smoke Tests

If the container is running, the quickest validation path is:

```bash
docker exec -it cd-manager /bin/bash -lc \
  "source activate cd-env && cd /causal-discovery && pytest"
```

The tests cover:

- graph state and cycle-handling logic
- ordering decoder behavior
- smoke runs for the UCT agent variants

## Running One Causal Discovery Job

The simplest entry point is `code/causal-discovery/cdrl/run_causal_discovery.py`.

It expects dataset and ground-truth paths relative to `/experiment_data`, plus:

- `--edge_budget`: maximum number of construction steps; this is not inferred automatically
- `--reg_type`: one of `LR`, `QR`, `GPR`
- `--score_type`: `BIC_different_var` (default) or `BIC`

Example: compare all three algorithms on Sachs and let ordering-UCT define the time budget used by the other two:

```bash
docker exec -it cd-manager /bin/bash -lc '
  source activate cd-env &&
  cd /causal-discovery &&
  python cdrl/run_causal_discovery.py \
    --dataset_file datasets/sachs/data.npy \
    --gt_file datasets/sachs/DAG.npy \
    --output_directory development/sachs_all_algorithms \
    --edge_budget 49 \
    --reg_type GPR \
    --algorithm all \
    --include_cam_pruning \
    --time_budget_s 0
'
```

Key flags:

- `--algorithm uct|ordering_uct|random_ordering|both|all`
- `--time_budget_s -1`: no wall-clock limit
- `--time_budget_s > 0`: fixed wall-clock budget per agent
- `--time_budget_s 0`: auto mode; only meaningful with `both` or `all`
- `--normalize_data`: enabled by default
- `--include_cam_pruning`: run CAM-based edge pruning after construction

Outputs are written under `/experiment_data/<output_directory>/` on the container side, which maps to `experiments/causal-discovery/<output_directory>/` in this repository.

Single-algorithm runs write:

- `<dataset>_results.json`
- `<dataset>_discovered_graph.pdf`
- `<dataset>_discovered_graph_pruned.pdf` if CAM pruning is enabled

Multi-algorithm runs write:

- `<dataset>_results_comparison.json`

## Task-Based Experiment Pipeline

The repository also keeps the batch experiment pipeline used for Sachs and synthetic scaling runs:

- `cdrl/setup_experiments.py`: create serialized `hyperopt` or `eval` tasks
- `cdrl/tasks.py`: execute one task
- `scripts/recipes/sachs_budget_experiment.sh`: helper for the Sachs comparison runs
- `scripts/recipes/time_budget_scaling_experiment.sh`: helper for the synthetic scaling runs

At a high level the flow is:

1. Create tasks with `setup_experiments.py`.
2. Execute `tasks.py` repeatedly for each generated task id.
3. Read metrics from `models/hyperopt_results/` and `models/eval_results/`.

Example: create the Sachs ordering-vs-random hyperparameter tasks:

```bash
docker exec -it cd-manager /bin/bash -lc '
  source activate cd-env &&
  cd /causal-discovery &&
  python cdrl/setup_experiments.py \
    --experiment_part hyperopt \
    --which sachs \
    --experiment_id sachs_budget_modifier_50_ordering_vs_random \
    --instance_name sachs \
    --agent_subset ordering_vs_random \
    --budget_modifier 50
'
```

Then execute a task:

```bash
docker exec -it cd-manager /bin/bash -lc '
  source activate cd-env &&
  cd /causal-discovery &&
  python cdrl/tasks.py \
    --experiment_id sachs_budget_modifier_50_ordering_vs_random \
    --experiment_part hyperopt \
    --task_id 1
'
```

Experiment directories follow the layout defined in `cdrl/io/file_paths.py`, with subdirectories such as:

- `models/tasks_hyperopt/`
- `models/tasks_eval/`
- `models/hyperopt_results/`
- `models/eval_results/`
- `figures/`
- `trajectories_data/`

## Checked-In Results And Plotting

This repository already includes result directories under `experiments/causal-discovery/`, for example:

- `sachs_budget_modifier_50_ordering_vs_random/`
- `sachs_time_budget_900s/`
- `synth10lr_time_budget_scaling/` through `synth50lr_time_budget_scaling/`

You can regenerate the comparison figures directly from those checked-in metrics without rerunning the experiments.

### Synthetic scaling comparison

```bash
python experiments/causal-discovery/plot_comparison.py --result-type prune_cam
```

This scans the `synth*lr_time_budget_scaling/` directories under `experiments/causal-discovery/` and writes:

- `experiments/causal-discovery/figures/comparison_construct.png`
- `experiments/causal-discovery/figures/comparison_prune_cam.png`

### Sachs comparison

Ordering-UCT vs random ordering:

```bash
python experiments/causal-discovery/plot_sachs_budget_modifier_comparison.py \
  --root experiments/causal-discovery \
  --algorithm-comparison ordering_vs_random \
  --result-type prune_cam
```

Ordering-UCT vs edge-based CD-UCT:

```bash
python experiments/causal-discovery/plot_sachs_budget_modifier_comparison.py \
  --root experiments/causal-discovery \
  --algorithm-comparison ordering_vs_uctfull \
  --result-type prune_cam
```

The script prints summary tables to the terminal and saves figures into the selected experiment directory's `figures/` folder.

## Practical Notes

- The code path that matters today is under `code/causal-discovery/cdrl/`.
- `experiments/causal-discovery/` is both an artifact store and a convenient mount point for `/experiment_data`.
- The repository includes historical environment files; expect some legacy pieces around the active ordering/CD-UCT implementation.
- The large experiment suites are still computationally expensive. The checked-in metrics and figures are the fastest way to inspect outcomes.
