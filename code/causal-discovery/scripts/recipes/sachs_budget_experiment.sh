#! /bin/bash
# Time-budget-fixed experiment on the Sachs dataset.
#
# Two modes controlled by the MODE env var:
#
#   uct_only (default):
#     OrderingMCTSAgent vs UCTFullDepthAgent (edge UCT), both running for the
#     same wall-clock time budget TIME_BUDGET_S.
#
#   ordering_vs_random:
#     OrderingMCTSAgent runs with a simulation budget (expansion_budget_modifier
#     = BUDGET_MODIFIER, default 200).  RandomOrderingBaseline then evaluates
#     exactly the same number of orderings that ordering UCT produced.
#     Hyperopt searches rollout_depth in {0, 2, 4}.
#
# Usage:
#   MODE=uct_only TIME_BUDGET_S=60 \
#     bash scripts/recipes/sachs_budget_experiment.sh [phase] [max_parallel_tasks]
#
#   phase: hyperopt | eval | both   (default: hyperopt)
#   TIME_BUDGET_S: wall-clock time budget in seconds (default: 60)
#   MODE: uct_only | ordering_vs_random   (default: uct_only)
#   BUDGET_MODIFIER: expansion_budget_modifier for ordering_uct in
#                    ordering_vs_random mode (default: 200)
#
# Requirements:
#   - CD_SOURCE_DIR must be set (e.g. export CD_SOURCE_DIR=/causal-discovery)
#   - CD_EXPERIMENT_DATA_DIR must be set
#   - Docker container cd-manager must be running

set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Usage: bash scripts/recipes/sachs_budget_experiment.sh [phase] [max_parallel_tasks]"
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

MODE="${MODE:-uct_only}"
if ! [[ "${MODE}" == "uct_only" || "${MODE}" == "ordering_vs_random" ]]; then
    echo "MODE must be one of: uct_only, ordering_vs_random."
    exit 1
fi

TIME_BUDGET_S="${TIME_BUDGET_S:-60}"
BUDGET_MODIFIER="${BUDGET_MODIFIER:-50}"

if [[ "${MODE}" == "uct_only" ]]; then
    echo "Using phase=${PHASE}, max_parallel_tasks=${MAX_PARALLEL_TASKS}, time_budget_s=${TIME_BUDGET_S}s, mode=${MODE}"
else
    echo "Using phase=${PHASE}, max_parallel_tasks=${MAX_PARALLEL_TASKS}, budget_modifier=${BUDGET_MODIFIER}, mode=${MODE}"
fi

INSTANCE="sachs"
if [[ "${MODE}" == "uct_only" ]]; then
    EXP_SUFFIX="time_budget_${TIME_BUDGET_S}s_${MODE}"
else
    EXP_SUFFIX="budget_modifier_${BUDGET_MODIFIER}_${MODE}"
fi
EXP_ID="${INSTANCE}_${EXP_SUFFIX}"

format_seconds() {
    local total_s=$1
    local hh=$((total_s / 3600))
    local mm=$(((total_s % 3600) / 60))
    local ss=$((total_s % 60))
    printf "%02d:%02d:%02d" "${hh}" "${mm}" "${ss}"
}

run_experiment_part() {
    local exp_part=$1

    local start_ts
    start_ts=$(date +%s)
    if [[ "${MODE}" == "uct_only" ]]; then
        echo ">>> ${exp_part} | ${INSTANCE} | time_budget_s=${TIME_BUDGET_S}s | mode=${MODE}"
    else
        echo ">>> ${exp_part} | ${INSTANCE} | budget_modifier=${BUDGET_MODIFIER} | mode=${MODE}"
    fi

    docker exec -it cd-manager /bin/bash -c \
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && \
         source activate cd-env && python /causal-discovery/cdrl/setup_experiments.py \
            --experiment_part ${exp_part} \
            --which sachs \
            --experiment_id ${EXP_ID} \
            --instance_name ${INSTANCE} \
            --time_budget_s ${TIME_BUDGET_S} \
            --agent_subset ${MODE} \
            --budget_modifier ${BUDGET_MODIFIER}"

    local task_count_file="${CD_EXPERIMENT_DATA_DIR}/${EXP_ID}/models/${exp_part}_tasks.count"
    local task_count
    task_count=$(cat "${task_count_file}" | tr -d '\n')
    echo "    ${exp_part} tasks created: ${task_count}"

    local next_task=1
    local completed=0
    local status_interval_s=10
    local last_status_ts=0
    declare -A pid_to_task=()

    while (( next_task <= task_count || ${#pid_to_task[@]} > 0 )); do
        while (( next_task <= task_count && ${#pid_to_task[@]} < MAX_PARALLEL_TASKS )); do
            docker exec cd-manager /bin/bash -c \
                "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && \
                 source activate cd-env && python /causal-discovery/cdrl/tasks.py \
                     --experiment_id ${EXP_ID} \
                     --experiment_part ${exp_part} \
                     --task_id ${next_task}" >/dev/null 2>&1 &
            local pid=$!
            pid_to_task["${pid}"]=${next_task}
            next_task=$((next_task + 1))
        done

        for pid in "${!pid_to_task[@]}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                wait "${pid}" || true
                unset 'pid_to_task[$pid]'
                completed=$((completed + 1))
            fi
        done

        local now_ts
        now_ts=$(date +%s)
        if (( now_ts - last_status_ts >= status_interval_s )) || (( completed == task_count )); then
            echo "    [${EXP_ID}:${exp_part}] completed ${completed}/${task_count} | ongoing ${#pid_to_task[@]}"
            last_status_ts=${now_ts}
        fi

        if (( completed < task_count )); then
            sleep 1
        fi
    done

    local end_ts
    end_ts=$(date +%s)
    echo "    Completed ${task_count} ${exp_part} tasks in $(format_seconds "$((end_ts - start_ts))")."
}

if [[ "${PHASE}" == "hyperopt" || "${PHASE}" == "both" ]]; then
    run_experiment_part hyperopt
    echo "Hyperopt complete."
fi

if [[ "${PHASE}" == "eval" || "${PHASE}" == "both" ]]; then
    run_experiment_part eval
    echo "Eval complete."
fi
