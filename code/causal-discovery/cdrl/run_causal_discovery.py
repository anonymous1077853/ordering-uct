import argparse
import json
import pprint
import sys
import time
from pathlib import Path

import networkx as nx


d = Path(__file__).resolve().parents[1]
sys.path.append(str(d.absolute()))

from cdrl.utils.general_utils import NpEncoder
from cdrl.state.dag_state import DAGState
from cdrl.utils.graph_utils import edge_list_to_nx_graph
from cdrl.agent.mcts.mcts_agent import MonteCarloTreeSearchAgent
from cdrl.agent.ordering.ordering_mcts_agent import OrderingMCTSAgent
from cdrl.agent.baseline.random_ordering_baseline import RandomOrderingBaseline
from cdrl.environment.graph_edge_env import DirectedGraphEdgeEnv, EnvPhase
from cdrl.evaluation.eval_utils import get_metrics_dict
from cdrl.reward_functions.reward_continuous_vars import ContinuousVarsBICRewardFunction
from cdrl.state.instance_generators import DiscoveryInstance, InstanceMetadata


def _run_single_agent(agent, disc_inst, opts, hyperparams, include_cam_pruning):
    """Run eval for a single agent and return the metrics dict and construct_output."""
    agent.setup(opts, hyperparams)
    construct_output = agent.eval([disc_inst.start_state], EnvPhase.CONSTRUCT)
    prune_output = None
    rfun = agent.environment.reward_function
    eval_dict = get_metrics_dict(
        construct_output, prune_output, disc_inst, rfun,
        include_cam_pruning=include_cam_pruning,
    )
    agent.finalize()
    return eval_dict, construct_output


def run_causal_discovery(args):
    dataset_path = Path("/experiment_data", args.dataset_file)

    gt_known = (args.gt_file is not None)

    gt_path = Path("/experiment_data", args.gt_file) if gt_known else None

    inst_name = dataset_path.stem

    instance_metadata = InstanceMetadata(name=inst_name, rvar_type="continuous", transpose=False, root_path=dataset_path.parent, reg_type=args.reg_type, rlbic_num_edges=-1)
    disc_inst = DiscoveryInstance(instance_metadata,
                                  data_path=str(dataset_path),
                                  dag_path=str(gt_path) if gt_known else None,
                                  normalize_data=args.normalize_data,
                                  starting_graph_generation="scratch",
                                  subsample_data=False
                                  )
    rfun = ContinuousVarsBICRewardFunction(disc_inst, penalise_cyclic=True, score_type=args.score_type, store_scores=True, reg_type=args.reg_type)

    initial_edge_budgets = {
        "construct": args.edge_budget,
        "prune": 0
    }

    env = DirectedGraphEdgeEnv(disc_inst, rfun, initial_edge_budgets=initial_edge_budgets, enforce_acyclic=True)

    opts = {
        "random_seed": args.random_seed,
    }

    include_cam_pruning = args.include_cam_pruning
    output_dir = Path("/experiment_data", args.output_directory)
    output_dir.mkdir(exist_ok=True, parents=True)

    run_uct = args.algorithm in ("uct", "both", "all")
    run_ordering = args.algorithm in ("ordering_uct", "both", "all")
    run_random_ordering = args.algorithm in ("random_ordering", "all")

    # When time_budget_s=0 (auto) and ordering-UCT is running alongside other agents:
    # ordering-UCT runs first to completion (as the reference agent), its wall-clock time
    # is measured, and that becomes the time budget for edge-UCT and random-ordering.
    # This lets us give all other agents exactly as much compute time as ordering-UCT needed.
    auto_time_budget = args.time_budget_s == 0 and run_ordering and (run_uct or run_random_ordering)
    # effective_time_budget starts at the explicit value; replaced after ordering-UCT runs in auto mode.
    effective_time_budget = -1 if auto_time_budget else args.time_budget_s

    combined_results = {}
    # Populated after ordering-UCT runs; passed as max_evals to baselines for budget parity.
    ordering_num_evals = None

    # Step 1: Ordering-based UCT (always first in auto mode; normal position otherwise)
    if run_ordering:
        ordering_agent = OrderingMCTSAgent(env)
        ordering_hyperparams = {
            'expansion_budget_modifier': args.expansion_budget_modifier,
            'exploration_c': args.exploration_c,
            'rollout_policy': args.ordering_rollout_policy,
            'rollout_depth': args.ordering_rollout_depth,
            'max_indegree': args.max_indegree,
            'coeff_threshold': args.coeff_threshold,
            'btm': args.btm,
            # Ordering-UCT always runs to completion (it is the reference); pass the
            # explicit budget only when auto mode is NOT active.
            'time_budget_s': -1 if auto_time_budget else effective_time_budget,
        }
        ordering_start = time.time()
        ordering_eval_dict, ordering_construct_output = _run_single_agent(
            ordering_agent, disc_inst, opts, ordering_hyperparams, include_cam_pruning
        )
        ordering_wall_clock_s = time.time() - ordering_start
        ordering_num_evals = ordering_agent.num_simulations

        if auto_time_budget:
            effective_time_budget = ordering_wall_clock_s
            print(f"[auto time budget] ordering-UCT took {ordering_wall_clock_s:.2f}s - "
                  f"using this as the budget for remaining agents.")

        del ordering_eval_dict["instance_metadata"]
        ordering_eval_dict["experiment_parameters"] = vars(args)
        ordering_eval_dict["wall_clock_s"] = ordering_wall_clock_s
        combined_results["ordering_uct"] = ordering_eval_dict

        if not run_uct and not run_random_ordering:
            # Single-agent output: flat format.
            results_file = output_dir / f"{inst_name}_results.json"
            with open(results_file, "w") as fh:
                json.dump(ordering_eval_dict, fh, indent=4, sort_keys=True, cls=NpEncoder)

            drawing_file = output_dir / f"{inst_name}_discovered_graph.pdf"
            final_state = ordering_construct_output[0][0]
            final_state.draw_to_file(drawing_file)

            if include_cam_pruning:
                drawing_file_cam = output_dir / f"{inst_name}_discovered_graph_pruned.pdf"
                post_pruning_edges = ordering_eval_dict['results']['prune_cam']['edges']
                post_pruning_state = DAGState(edge_list_to_nx_graph(post_pruning_edges, disc_inst.d))
                post_pruning_state.draw_to_file(drawing_file_cam)

            print("=" * 50)
            print("Final results after construction phase (CD-Ordering-UCT):")
            print("=" * 50)
            print_results_to_console(ordering_eval_dict['results']['construct'], gt_known)
            if include_cam_pruning:
                print("=" * 50)
                print("Final results after CAM pruning phase (CD-Ordering-UCT):")
                print("=" * 50)
                print_results_to_console(ordering_eval_dict['results']['prune_cam'], gt_known)

            print(f"Wrote causal discovery results to file {results_file.resolve()}.")
            return

    # Step 2: Edge-based UCT
    if run_uct:
        uct_agent = MonteCarloTreeSearchAgent(env)
        uct_hyperparams = {
            'C_p': args.C_p,
            'adjust_C_p': args.adjust_C_p,
            'final_action_strategy': args.final_action_strategy,
            'expansion_budget_modifier': args.expansion_budget_modifier,
            'sim_policy': args.sim_policy,
            'rollout_depth': args.rollout_depth,
            'sims_per_expansion': args.sims_per_expansion,
            'transpositions_enabled': args.transpositions_enabled,
            'btm': args.btm,
            'time_budget_s': effective_time_budget,
        }
        uct_eval_dict, uct_construct_output = _run_single_agent(
            uct_agent, disc_inst, opts, uct_hyperparams, include_cam_pruning
        )

        del uct_eval_dict["instance_metadata"]
        uct_eval_dict["experiment_parameters"] = vars(args)
        combined_results["uct"] = uct_eval_dict

        if not run_ordering and not run_random_ordering:
            # Single-agent output: write in the original flat format for backward compatibility.
            results_file = output_dir / f"{inst_name}_results.json"
            with open(results_file, "w") as fh:
                json.dump(uct_eval_dict, fh, indent=4, sort_keys=True, cls=NpEncoder)

            drawing_file = output_dir / f"{inst_name}_discovered_graph.pdf"
            final_state = uct_construct_output[0][0]
            final_state.draw_to_file(drawing_file)

            if include_cam_pruning:
                drawing_file_cam = output_dir / f"{inst_name}_discovered_graph_pruned.pdf"
                post_pruning_edges = uct_eval_dict['results']['prune_cam']['edges']
                post_pruning_state = DAGState(edge_list_to_nx_graph(post_pruning_edges, disc_inst.d))
                post_pruning_state.draw_to_file(drawing_file_cam)

            print("=" * 50)
            print("Final results after construction phase (CD-UCT):")
            print("=" * 50)
            print_results_to_console(uct_eval_dict['results']['construct'], gt_known)
            if include_cam_pruning:
                print("=" * 50)
                print("Final results after CAM pruning phase (CD-UCT):")
                print("=" * 50)
                print_results_to_console(uct_eval_dict['results']['prune_cam'], gt_known)

            print(f"Wrote causal discovery results to file {results_file.resolve()}.")
            return

    # Step 3: Random ordering baseline
    if run_random_ordering:
        random_ordering_agent = RandomOrderingBaseline(env)
        random_ordering_hyperparams = {
            'expansion_budget_modifier': args.expansion_budget_modifier,
            'max_indegree': args.max_indegree,
            'coeff_threshold': args.coeff_threshold,
            'time_budget_s': effective_time_budget,
            'max_evals': ordering_num_evals,  # None when UCT didn't run -> falls back to formula
        }
        random_ordering_eval_dict, random_ordering_construct_output = _run_single_agent(
            random_ordering_agent, disc_inst, opts, random_ordering_hyperparams, include_cam_pruning
        )

        del random_ordering_eval_dict["instance_metadata"]
        random_ordering_eval_dict["experiment_parameters"] = vars(args)
        combined_results["random_ordering"] = random_ordering_eval_dict

        if not run_uct and not run_ordering:
            # Single-agent output: flat format.
            results_file = output_dir / f"{inst_name}_results.json"
            with open(results_file, "w") as fh:
                json.dump(random_ordering_eval_dict, fh, indent=4, sort_keys=True, cls=NpEncoder)

            drawing_file = output_dir / f"{inst_name}_discovered_graph.pdf"
            final_state = random_ordering_construct_output[0][0]
            final_state.draw_to_file(drawing_file)

            if include_cam_pruning:
                drawing_file_cam = output_dir / f"{inst_name}_discovered_graph_pruned.pdf"
                post_pruning_edges = random_ordering_eval_dict['results']['prune_cam']['edges']
                post_pruning_state = DAGState(edge_list_to_nx_graph(post_pruning_edges, disc_inst.d))
                post_pruning_state.draw_to_file(drawing_file_cam)

            print("=" * 50)
            print("Final results after construction phase (Random-Ordering Baseline):")
            print("=" * 50)
            print_results_to_console(random_ordering_eval_dict['results']['construct'], gt_known)
            if include_cam_pruning:
                print("=" * 50)
                print("Final results after CAM pruning phase (Random-Ordering Baseline):")
                print("=" * 50)
                print_results_to_console(random_ordering_eval_dict['results']['prune_cam'], gt_known)

            print(f"Wrote causal discovery results to file {results_file.resolve()}.")
            return

    # Multiple agents: write combined output
    results_file = output_dir / f"{inst_name}_results_comparison.json"
    with open(results_file, "w") as fh:
        json.dump(combined_results, fh, indent=4, sort_keys=True, cls=NpEncoder)

    if run_uct:
        print("=" * 50)
        print("CD-UCT (edge-based):")
        print("=" * 50)
        print_results_to_console(combined_results["uct"]['results']['construct'], gt_known)
        if include_cam_pruning:
            print("  After CAM pruning:")
            print_results_to_console(combined_results["uct"]['results']['prune_cam'], gt_known)

    if run_ordering:
        print("=" * 50)
        print("CD-Ordering-UCT (ordering-based):")
        print("=" * 50)
        print_results_to_console(combined_results["ordering_uct"]['results']['construct'], gt_known)
        if include_cam_pruning:
            print("  After CAM pruning:")
            print_results_to_console(combined_results["ordering_uct"]['results']['prune_cam'], gt_known)

    if run_random_ordering:
        print("=" * 50)
        print("Random-Ordering Baseline:")
        print("=" * 50)
        print_results_to_console(combined_results["random_ordering"]['results']['construct'], gt_known)
        if include_cam_pruning:
            print("  After CAM pruning:")
            print_results_to_console(combined_results["random_ordering"]['results']['prune_cam'], gt_known)

    print(f"Wrote comparison results to file {results_file.resolve()}.")


def print_results_to_console(results_dict, gt_known):
    print(f"Score Function Value: \t\t\t{results_dict['reward']:.3f}")
    print(f"Number of Edges: \t\t\t{results_dict['pred_size']}")
    if gt_known:
        print(f"True Positive Rate (TPR): \t\t{results_dict['tpr']:.3f}")
        print(f"False Discovery Rate (FDR): \t\t{results_dict['fdr']:.3f}")
        print(f"Structural Hamming Distance (SHD): \t{results_dict['shd']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run causal discovery algorithm for a specified dataset.")
    parser.add_argument("--dataset_file", required=True, type=str,
                        help="Path to dataset file relative to $CD_EXPERIMENT_DATA_DIR. Can be in .csv or .npy format.")

    parser.add_argument("--gt_file", required=False, type=str,
                        help="Path to file containing the adjacency matrix of the ground truth graph relative to $CD_EXPERIMENT_DATA_DIR (optional). Can be in .csv or .npy format.")

    parser.add_argument("--output_directory", required=True, type=str,
                        help="Path to output directory relative to $CD_EXPERIMENT_DATA_DIR.")

    parser.add_argument('--normalize_data', action='store_true', help="Whether to normalize data before running regression.")
    parser.set_defaults(normalize_data=True)

    parser.add_argument("--edge_budget", type=int, required=True, help="Edge budget for the causal discovery problem.")

    parser.add_argument("--reg_type", type=str, required=True, help="Type of regression to run: linear regression (LR), quadratic regression (QR), or Gaussian process regression (GPR).", choices=["LR", "QR", "GPR"])
    parser.add_argument("--score_type", type=str, required=False, help="Score function: BIC with heterogeneous (BIC_different_var) or equal (BIC) variances", choices=["BIC_different_var", "BIC"], default="BIC_different_var")

    parser.add_argument('--include_cam_pruning', action='store_true', help="Whether to apply pruning of edges via statistical tests with the CAM method after the construction phase.")
    parser.set_defaults(include_cam_pruning=False)

    parser.add_argument('--random_seed', type=int, help="Random seed to use as initialization to random number generator.", default=100)

    # Algorithm selection
    parser.add_argument("--algorithm", type=str, required=False,
                        help="Which algorithm to run: 'uct' (edge-based MCTS), 'ordering_uct' (ordering-based MCTS), 'random_ordering' (random ordering baseline), 'both' (uct + ordering_uct), or 'all' (all three).",
                        choices=["uct", "ordering_uct", "random_ordering", "both", "all"], default="uct")

    # Shared budget parameter (used by both agents)
    parser.add_argument("--expansion_budget_modifier", type=float, required=False,
                        help="Simulation budget modifier. Edge-MCTS: expansions per timestep = d * modifier. Ordering-MCTS: simulations per ordering step = d * modifier.",
                        default=25)

    # Edge-based UCT (CD-UCT) hyperparameters
    parser.add_argument("--C_p", type=float, required=False, help="CD-UCT exploration parameter.", default=0.025)
    parser.add_argument('--adjust_C_p', action='store_true', help="Whether to adjust the C_p parameter depending on the Q-value at the root. Recommended to leave on.")
    parser.set_defaults(adjust_C_p=True)

    parser.add_argument("--final_action_strategy", type=str, required=False, help="CD-UCT final mechanism for child action selection .", choices=["max_child", "robust_child"], default="robust_child")
    parser.add_argument("--sim_policy", type=str, required=False, help="CD-UCT simulation policy.", choices=["random", "naive"], default="random")
    parser.add_argument("--rollout_depth", type=int, required=False, help="CD-UCT rollout depth in terms of number of edges. -1 corresponds to full rollouts (i.e., until the end of the MDP).", default=-1)
    parser.add_argument("--sims_per_expansion", type=int, required=False, help="CD-UCT number of simulations per expansion.", default=1)

    parser.add_argument('--transpositions_enabled', action='store_true', help="Whether to use transpositions when performing the search. Experimental feature. Recommended to leave off.")
    parser.set_defaults(transpositions_enabled=False)

    parser.add_argument('--btm', action='store_true', help="Whether to memorize the best trajectory encountered during the search. Recommended to leave on.")
    parser.set_defaults(btm=True)

    # Ordering-based UCT (CD-Ordering-UCT) hyperparameters
    parser.add_argument("--exploration_c", type=float, required=False,
                        help="CD-Ordering-UCT UCT exploration constant (replaces C_p; reward normalisation makes this scale-invariant).",
                        default=1.4)
    parser.add_argument("--ordering_rollout_policy", type=str, required=False,
                        help="CD-Ordering-UCT rollout policy: 'random' (fast) or 'greedy' (guided by local BIC score).",
                        choices=["random", "greedy"], default="random")
    parser.add_argument("--ordering_rollout_depth", type=int, required=False,
                        help="CD-Ordering-UCT rollout depth: 0 = full rollout to terminal; k > 0 = extend by k variables.",
                        default=0)
    parser.add_argument("--max_indegree", type=int, required=False,
                        help="CD-Ordering-UCT decoder Stage-2 max parents per node. Use -1 (default) for no cap: all ordering-consistent predecessors are candidate parents.",
                        default=-1)
    parser.add_argument("--coeff_threshold", type=float, required=False,
                        help="CD-Ordering-UCT decoder Stage-3 LR/QR coefficient pruning threshold.",
                        default=0.3)

    # Wall-clock time budget (optional)
    parser.add_argument("--time_budget_s", type=float, required=False,
                        help=(
                            "Wall-clock time budget in seconds. "
                            "-1 (default): no limit - use simulation count as normal. "
                            ">0: each agent stops after this many seconds. "
                            "0 (auto): ordering-UCT runs to completion first; its elapsed time "
                            "becomes the budget given to edge-UCT and random-ordering (requires "
                            "--algorithm both/all)."
                        ),
                        default=-1)

    args = parser.parse_args()
    run_causal_discovery(args)


if __name__ == "__main__":
    main()
