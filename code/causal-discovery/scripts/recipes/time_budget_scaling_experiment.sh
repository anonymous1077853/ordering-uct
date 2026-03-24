#! /bin/bash
# Time-budget-fixed scaling experiment.
#
# Both CD-UCT (edge-based) and CD-Ordering-UCT are given the same wall-clock
# time budget T(d) per run, where T(d) is calibrated by running ordering-UCT
# once with the standard modifier (25) and measuring its elapsed time.
#
# Calibration -> Hyperopt -> Eval pipeline:
#
#   Step 0 (calibration): fill in TIME_BUDGETS below by running
#       python cdrl/run_causal_discovery.py --algorithm ordering_uct \
#           --expansion_budget_modifier 25 ...
#   for each instance and noting the wall_clock_s printed in the results JSON.
#
#   Step 1 (hyperopt): both agents search over their hyperparameter grids
#       subject to time_budget_s seconds per seed.
#
#   Step 2 (eval): both agents are evaluated using the best hyperparams found
#       in Step 1, still respecting the same time budget.
#
# Usage:
#   AGENT_SUBSET=ordering bash scripts/recipes/time_budget_scaling_experiment.sh [phase] [max_parallel_tasks]
#   phase: hyperopt | eval | both   (default: hyperopt)
#   AGENT_SUBSET: both | ordering | edge_uct (default: both)
#
# Requirements:
#   - CD_SOURCE_DIR must be set (e.g. export CD_SOURCE_DIR=/causal-discovery)
#   - CD_EXPERIMENT_DATA_DIR must be set
#   - Docker container cd-manager must be running

set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Usage: bash scripts/recipes/time_budget_scaling_experiment.sh [phase] [max_parallel_tasks]"
    exit 1
fi

PHASE="hyperopt"
MAX_PARALLEL_TASKS=""

if [[ $# -ge 1 ]]; then
    if [[ "${1}" =~ ^[1-9][0-9]*$ ]]; then
        MAX_PARALLEL_TASKS="${1}"
    else
        PHASE="${1}"
    fi
fi

if [[ $# -eq 2 ]]; then
    if [[ "${2}" =~ ^[1-9][0-9]*$ ]]; then
        MAX_PARALLEL_TASKS="${2}"
    else
        echo "max_parallel_tasks must be a positive integer."
        exit 1
    fi
fi

if ! [[ "${PHASE}" == "hyperopt" || "${PHASE}" == "eval" || "${PHASE}" == "both" ]]; then
    echo "phase must be one of: hyperopt, eval, both."
    exit 1
fi

if [[ -z "${MAX_PARALLEL_TASKS}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        MAX_PARALLEL_TASKS="$(nproc)"
    else
        MAX_PARALLEL_TASKS=1
    fi
fi

echo "Using phase=${PHASE}, max_parallel_tasks=${MAX_PARALLEL_TASKS}"

AGENT_SUBSET="${AGENT_SUBSET:-both}"
if ! [[ "${AGENT_SUBSET}" == "both" || "${AGENT_SUBSET}" == "ordering" || "${AGENT_SUBSET}" == "edge_uct" ]]; then
    echo "AGENT_SUBSET must be one of: both, ordering, edge_uct."
    exit 1
fi
echo "Using AGENT_SUBSET=${AGENT_SUBSET}"

# Instances
instances=(
    "synth10lr"
    "synth15lr"
    "synth20lr"
    "synth25lr"
    "synth30lr"
    "synth35lr"
    "synth40lr"
    "synth45lr"
    "synth50lr"
)

# Time budgets (seconds) from calibration
# Fill these values by running ordering-UCT with expansion_budget_modifier=25
# on each instance and reading wall_clock_s from the output JSON.
# Placeholder values below - replace with machine-specific measurements.
declare -A TIME_BUDGETS=(
    [synth10lr]=300
    [synth15lr]=300
    [synth20lr]=300
    [synth25lr]=300
    [synth30lr]=300
    [synth35lr]=300
    [synth40lr]=300
    [synth45lr]=300
    [synth50lr]=300
)

EXP_SUFFIX="time_budget_scaling"
if [[ "${AGENT_SUBSET}" != "both" ]]; then
    EXP_SUFFIX="${EXP_SUFFIX}_${AGENT_SUBSET}"
fi
RUN_EXPERIMENT_LAST_ELAPSED_S=0

format_seconds() {
    local total_s=$1
    local hh=$((total_s / 3600))
    local mm=$(((total_s % 3600) / 60))
    local ss=$((total_s % 60))
    printf "%02d:%02d:%02d" "${hh}" "${mm}" "${ss}"
}

ceil_div() {
    local a=$1
    local b=$2
    echo $(((a + b - 1) / b))
}

# Helper: setup + launch tasks for one experiment part
run_experiment_part() {
    local exp_part=$1      # "hyperopt" or "eval"
    local instance=$2
    local T=$3

    local exp_id="${instance}_${EXP_SUFFIX}"
    local start_ts
    start_ts=$(date +%s)

    echo ">>> ${exp_part} | ${instance} | T=${T}s"

    docker exec -it cd-manager /bin/bash -c \
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && \
         source activate cd-env && python /causal-discovery/cdrl/setup_experiments.py \
            --experiment_part ${exp_part} \
            --which time_budget \
            --experiment_id ${exp_id} \
            --instance_name ${instance} \
            --budget 1000000 \
            --agent_subset ${AGENT_SUBSET} \
            --time_budget_s ${T}"

    local task_count_file="${CD_EXPERIMENT_DATA_DIR}/${exp_id}/models/${exp_part}_tasks.count"
    local task_count
    task_count=$(cat "${task_count_file}" | tr -d '\n')
    local planned_upper_s
    planned_upper_s=$(( "$(ceil_div "${task_count}" "${MAX_PARALLEL_TASKS}")" * T ))
    echo "    Planned budget upper bound for this instance: $(format_seconds "${planned_upper_s}")"

    local next_task=1
    local completed=0
    local status_interval_s=10
    local last_status_ts=0
    declare -A pid_to_task=()
    declare -A pid_to_start=()

    while (( next_task <= task_count || ${#pid_to_task[@]} > 0 )); do
        while (( next_task <= task_count && ${#pid_to_task[@]} < MAX_PARALLEL_TASKS )); do
            docker exec cd-manager /bin/bash -c \
                "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && \
                 source activate cd-env && python /causal-discovery/cdrl/tasks.py \
                     --experiment_id ${exp_id} \
                     --experiment_part ${exp_part} \
                     --task_id ${next_task}" >/dev/null 2>&1 &
            local pid=$!
            pid_to_task["${pid}"]=${next_task}
            pid_to_start["${pid}"]=$(date +%s)
            next_task=$((next_task + 1))
        done

        # Reap completed background jobs.
        for pid in "${!pid_to_task[@]}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                if wait "${pid}"; then
                    :
                fi
                unset 'pid_to_task[$pid]'
                unset 'pid_to_start[$pid]'
                completed=$((completed + 1))
            fi
        done

        local now_ts
        now_ts=$(date +%s)
        if (( now_ts - last_status_ts >= status_interval_s )) || (( completed == task_count )); then
            local ongoing_count=${#pid_to_task[@]}
            local launched=$((next_task - 1))
            local remaining_to_launch=$((task_count - launched))

            local sum_active_remaining=0
            for pid in "${!pid_to_task[@]}"; do
                local started_at=${pid_to_start[$pid]}
                local elapsed_task=$((now_ts - started_at))
                local remaining_task=$((T - elapsed_task))
                if (( remaining_task < 0 )); then
                    remaining_task=0
                fi
                sum_active_remaining=$((sum_active_remaining + remaining_task))
            done

            local total_remaining_budget_s=$((sum_active_remaining + remaining_to_launch * T))
            local eta_s=$(( "$(ceil_div "${total_remaining_budget_s}" "${MAX_PARALLEL_TASKS}")" ))

            echo "    [${exp_id}:${exp_part}] instance time left: $(format_seconds "${eta_s}") | completed ${completed}/${task_count} | ongoing ${ongoing_count}"
            last_status_ts=${now_ts}
        fi

        if (( completed < task_count )); then
            sleep 1
        fi
    done

    local end_ts
    end_ts=$(date +%s)
    local elapsed_s=$((end_ts - start_ts))
    RUN_EXPERIMENT_LAST_ELAPSED_S=${elapsed_s}
    echo "    Completed ${task_count} ${exp_part} tasks for ${instance} in $(format_seconds "${elapsed_s}")."
}

# Main loop
if [[ "${PHASE}" == "hyperopt" || "${PHASE}" == "both" ]]; then
    total_instances=${#instances[@]}
    completed_instances=0
    phase_start_ts=$(date +%s)

    for instance in "${instances[@]}"; do
        completed_instances=$((completed_instances + 1))
        echo "[hyperopt ${completed_instances}/${total_instances}] starting ${instance}"

        T="${TIME_BUDGETS[${instance}]}"
        # Step 1: Hyperparameter optimisation - both agents, same time budget
        run_experiment_part hyperopt "${instance}" "${T}"

        now_ts=$(date +%s)
        phase_elapsed_s=$((now_ts - phase_start_ts))
        remaining_instances=$((total_instances - completed_instances))
        avg_per_instance_s=$((phase_elapsed_s / completed_instances))
        phase_eta_s=$((avg_per_instance_s * remaining_instances))
        echo "    [hyperopt progress] completed ${completed_instances}/${total_instances} instances | elapsed $(format_seconds "${phase_elapsed_s}") | eta $(format_seconds "${phase_eta_s}")"
    done
    echo ""
    echo "All hyperopt tasks completed."
    echo ""
fi

if [[ "${PHASE}" == "eval" || "${PHASE}" == "both" ]]; then
    total_instances=${#instances[@]}
    completed_instances=0
    phase_start_ts=$(date +%s)

    for instance in "${instances[@]}"; do
        completed_instances=$((completed_instances + 1))
        echo "[eval ${completed_instances}/${total_instances}] starting ${instance}"

        T="${TIME_BUDGETS[${instance}]}"
        # Step 2: Evaluation using best hyperparams from hyperopt
        run_experiment_part eval "${instance}" "${T}"

        now_ts=$(date +%s)
        phase_elapsed_s=$((now_ts - phase_start_ts))
        remaining_instances=$((total_instances - completed_instances))
        avg_per_instance_s=$((phase_elapsed_s / completed_instances))
        phase_eta_s=$((avg_per_instance_s * remaining_instances))
        echo "    [eval progress] completed ${completed_instances}/${total_instances} instances | elapsed $(format_seconds "${phase_elapsed_s}") | eta $(format_seconds "${phase_eta_s}")"
    done
    echo ""
    echo "All eval tasks completed."
    echo ""
fi
