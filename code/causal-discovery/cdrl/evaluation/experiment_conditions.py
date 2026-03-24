from copy import deepcopy

from cdrl.agent.mcts.mcts_agent import (
    UCTDepth2Agent,
    UCTDepth4Agent,
    UCTDepth8Agent,
    UCTDepth16Agent,
    UCTDepth32Agent,
    UCTFullDepthAgent,
)
from cdrl.agent.ordering.ordering_mcts_agent import OrderingMCTSAgent
from cdrl.agent.baseline.random_ordering_baseline import RandomOrderingBaseline
from cdrl.reward_functions.reward_continuous_vars import ContinuousVarsBICRewardFunction
from cdrl.state.instance_generators import HardcodedInstanceGenerator

known_cmd_kwargs = ["gt", "n", "p", "e", "what_vary", "agent_subset", "budget_modifier"]


class ExperimentConditions(object):
    """
    Base class that specifies the full configuration for an experiment.

    Subclasses (TimeBudgetExperimentConditions, SachsExperimentConditions) override
    get_agents(), create_hyperparam_grids(), and seed counts to match each experiment's
    needs.  setup_experiments.py instantiates the appropriate subclass and uses it to
    create serialized task objects for hyperopt and eval phases.

    Key attributes set by __init__:
        agents: list of agent classes to include in this experiment.
        test_seeds / validation_seeds: integer seeds for eval and hyperopt respectively.
        hyperparam_grids: nested dict {objective_name: {algorithm_name: {param: [values]}}}.
        btm_on_eval: whether to enable Best Trajectory Memory (BTM) during eval.
            Can be a bool (applies to all agents) or a dict keyed by algorithm_name
            (e.g. {"uctfull": False, "ordering_uct": True} when the sentinel
            expansion_budget_modifier breaks the edge-UCT BTM check).
        network_generators: list of instance generator classes (typically [HardcodedInstanceGenerator]).
        objective_functions: list of reward function classes (typically [ContinuousVarsBICRewardFunction]).
    """

    def __init__(self, instance_name, vars_dict):
        self.instance_name = instance_name

        for k, v in vars_dict.items():
            if k in known_cmd_kwargs:
                setattr(self, k, v)

        self.perform_construction = True
        self.perform_pruning = False
        self.starting_graph_generation = "scratch"

        self.btm_on_eval = True

        self.objective_functions = [ContinuousVarsBICRewardFunction]

        self.network_generators = [HardcodedInstanceGenerator]

        self.hyps_chunk_size = 1
        self.seeds_chunk_size = 1

        self.agents = self.get_agents()

    def get_agents(self):
        agent_subset = getattr(self, "agent_subset", "both")
        if agent_subset == "both":
            return [self.get_mcts_class(self.instance_name), OrderingMCTSAgent]
        if agent_subset == "ordering":
            return [OrderingMCTSAgent]
        if agent_subset == "edge_uct":
            return [self.get_mcts_class(self.instance_name)]
        raise ValueError(f"agent subset {agent_subset} not recognized.")

    def get_ordering_max_indegree(self):
        """
        Returns ordering decoder max_indegree for this instance.
        For synth*lr datasets, use the true graph max indegree; otherwise keep
        auto mode (-1).
        """
        true_max_indegree = {
            "sachs": 3,
            "synth10lr": 1,
            "synth15lr": 3,
            "synth20lr": 3,
            "synth25lr": 5,
            "synth30lr": 4,
            "synth35lr": 6,
            "synth40lr": 7,
            "synth45lr": 8,
            "synth50lr": 8,
        }
        return true_max_indegree.get(self.instance_name, -1)

    def get_mcts_class(self, instance_name):
        if instance_name == "sachs":
            return UCTFullDepthAgent
        if instance_name.startswith("syntren"):
            return UCTDepth8Agent
        if instance_name == "synth50qr":
            return UCTDepth4Agent
        return UCTFullDepthAgent


class MainExperimentConditions(ExperimentConditions):
    def __init__(self, instance_name, vars_dict):
        super().__init__(instance_name, vars_dict)

        self.experiment_params = {
            "num_runs": 20,  # number of random seeds
            # "score_type": "BIC",  # BIC with equal variances
            "score_type": "BIC_different_var",  # BIC with heterogeneous variances
            "enforce_acyclic": True,
            "penalise_cyclic": True,
        }

        self.num_seeds_skip = 0
        self.seed_offset = int(self.num_seeds_skip * 42)

        self.test_seeds = [
            self.seed_offset + int(run_number * 42)
            for run_number in range(self.experiment_params["num_runs"])
        ]
        self.validation_seeds = [
            self.seed_offset + int(run_number * 42)
            for run_number in range(
                len(self.test_seeds),
                self.experiment_params["num_runs"] + len(self.test_seeds),
            )
        ]

        self.hyperparam_grids = self.create_hyperparam_grids(vars_dict)

    def create_hyperparam_grids(self, vars_dict):
        if "budget" in vars_dict:
            budget_modifiers = [vars_dict["budget"]]
        else:
            budget_modifiers = [25]

        base_mcts = {
            "C_p": [0.025, 0.05, 0.075, 0.1],
            "expansion_budget_modifier": budget_modifiers,
            "sims_per_expansion": [1],
            "adjust_C_p": [True],
            "sim_policy": ["random"],
            "final_action_strategy": ["robust_child"],
            "transpositions_enabled": [False],
        }
        ordering_grid = {
            "expansion_budget_modifier": budget_modifiers,
            "exploration_c": [0.025, 0.05, 0.075, 0.1],
            "adjust_C_p": [True],
            "rollout_policy": ["random", "greedy"],
            "rollout_depth": [0, 1, 2, 3],
            "max_indegree": [self.get_ordering_max_indegree()],
            "coeff_threshold": [0.3],
        }
        hyperparam_grid_base = {
            UCTDepth2Agent.algorithm_name: deepcopy(base_mcts),
            UCTDepth4Agent.algorithm_name: deepcopy(base_mcts),
            UCTDepth8Agent.algorithm_name: deepcopy(base_mcts),
            UCTDepth16Agent.algorithm_name: deepcopy(base_mcts),
            UCTDepth32Agent.algorithm_name: deepcopy(base_mcts),
            UCTFullDepthAgent.algorithm_name: deepcopy(base_mcts),
            OrderingMCTSAgent.algorithm_name: ordering_grid,
        }

        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids

    def get_initial_edge_budgets(self, network_generator, discovery_instance):
        return {"construct": discovery_instance.true_num_edges, "prune": 0}


class TimeBudgetExperimentConditions(MainExperimentConditions):
    """
    Experiment conditions for the wall-clock-time-fixed scaling experiment.

    Both UCTFullDepthAgent and OrderingMCTSAgent are given the same time budget T
    (passed via vars_dict["time_budget_s"]).  expansion_budget_modifier is set to
    a very large sentinel so the time budget is always the binding constraint.

    Both agents search over their normal hyperparameter grids within that time budget
    ensuring a fair comparison: hyperopt and eval both respect the same T.
    """

    def __init__(self, instance_name, vars_dict):
        super().__init__(instance_name, vars_dict)
        # Override seed count to 20 for this experiment.
        self.experiment_params["num_runs"] = 20
        self.test_seeds = [
            self.seed_offset + int(run_number * 42)
            for run_number in range(self.experiment_params["num_runs"])
        ]
        self.validation_seeds = [
            self.seed_offset + int(run_number * 42)
            for run_number in range(
                len(self.test_seeds),
                self.experiment_params["num_runs"] + len(self.test_seeds),
            )
        ]
        # BTM is broken for edge UCT when using the sentinel budget: starting_budgets[i]=1M
        # can never equal final_state.num_edges for any realistic graph, so update_best_trajectories
        # never fires and edge UCT always falls back to the committed greedy path.
        # Ordering UCT's BTM check (len(sim_order)==d) is always evaluable and works correctly.
        self.btm_on_eval = {
            "uctfull": False,        # UCTFullDepthAgent - BTM broken with sentinel budget
            "ordering_uct": True,    # OrderingMCTSAgent - BTM works correctly
        }

    def get_agents(self):
        agent_subset = getattr(self, "agent_subset", "both")
        if agent_subset == "both":
            return [UCTFullDepthAgent, OrderingMCTSAgent]
        if agent_subset == "ordering":
            return [OrderingMCTSAgent]
        if agent_subset == "edge_uct":
            return [UCTFullDepthAgent]
        raise ValueError(f"agent subset {agent_subset} not recognized.")

    def create_hyperparam_grids(self, vars_dict):
        T = vars_dict.get("time_budget_s", -1)

        ordering_grid = {
            "expansion_budget_modifier": [1_000_000],
            "time_budget_s": [T],
            "exploration_c": [0.025, 0.05, 0.075, 0.1],
            # Keep exploration-rescaling behaviour aligned with edge CD-UCT.
            "adjust_C_p": [True],
            "rollout_policy": ["random"],
            "rollout_depth": [0],
            "max_indegree": [self.get_ordering_max_indegree()],
            "coeff_threshold": [0.3],
        }
        uct_grid = {
            "C_p": [0.025, 0.05, 0.075, 0.1],
            "expansion_budget_modifier": [1_000_000],
            "time_budget_s": [T],
            "sims_per_expansion": [1],
            "adjust_C_p": [True],
            "sim_policy": ["random"],
            "final_action_strategy": ["robust_child"],
            "transpositions_enabled": [False],
        }
        hyperparam_grid_base = {
            UCTFullDepthAgent.algorithm_name: uct_grid,
            OrderingMCTSAgent.algorithm_name: ordering_grid,
        }

        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids


class SachsExperimentConditions(MainExperimentConditions):
    """
    Experiment conditions for the Sachs experiment.  Two modes controlled by agent_subset:

    uct_only (default):
        OrderingMCTSAgent vs UCTFullDepthAgent, same wall-clock time budget T.
        expansion_budget_modifier sentinel (1M) so time budget is always binding.
        BTM disabled for edge UCT (sentinel breaks the BTM update check).

    ordering_vs_random:
        OrderingMCTSAgent runs with a simulation budget (expansion_budget_modifier =
        budget_modifier, default 200; time_budget_s = -1).  RandomOrderingBaseline
        then evaluates exactly ordering_uct.num_simulations simulations per seed
        (injected by SachsOrderingGroupEvalTask).
        Hyperopt searches rollout_depth in {0, 2, 4}.
    """

    def __init__(self, instance_name, vars_dict):
        super().__init__(instance_name, vars_dict)
        # Default budget_modifier if not supplied via CLI.
        if not hasattr(self, "budget_modifier"):
            self.budget_modifier = 50
        # Override seed count to 50 for this experiment.
        self.experiment_params["num_runs"] = 50
        self.test_seeds = [
            self.seed_offset + int(run_number * 42)
            for run_number in range(self.experiment_params["num_runs"])
        ]
        self.validation_seeds = [
            self.seed_offset + int(run_number * 42)
            for run_number in range(
                len(self.test_seeds),
                self.experiment_params["num_runs"] + len(self.test_seeds),
            )
        ]
        # BTM is broken for edge UCT when using the sentinel expansion_budget_modifier (1M):
        # starting_budgets[i]=1M never equals final_state.num_edges for any realistic graph,
        # so update_best_trajectories never fires.
        # Ordering UCT's BTM check (len(sim_order)==d) is always evaluable and works correctly.
        self.btm_on_eval = {
            "uctfull": False,
            "ordering_uct": True,
        }

    def get_agents(self):
        agent_subset = getattr(self, "agent_subset", "uct_only")
        if agent_subset == "uct_only":
            return [UCTFullDepthAgent, OrderingMCTSAgent]
        if agent_subset == "ordering_vs_random":
            return [OrderingMCTSAgent, RandomOrderingBaseline]
        if agent_subset == "edge_uct":
            return [UCTFullDepthAgent]
        raise ValueError(f"agent_subset {agent_subset!r} not recognised.")

    def create_hyperparam_grids(self, vars_dict):
        T = vars_dict.get("time_budget_s", -1)
        agent_subset = vars_dict.get("agent_subset", "uct_only")

        base_mcts = {
            "C_p": [0.025, 0.05, 0.075, 0.1],
            "expansion_budget_modifier": [1_000_000],
            "time_budget_s": [T],
            "sims_per_expansion": [1],
            "adjust_C_p": [True],
            "sim_policy": ["random"],
            "final_action_strategy": ["robust_child"],
            "transpositions_enabled": [False],
        }

        # coeff_threshold omitted: Sachs uses GPR, which dispatches to pruning_cam()
        # and ignores coeff_threshold entirely.
        if agent_subset == "ordering_vs_random":
            # Simulation-budget mode: ordering_uct runs budget_modifier sims per step,
            # then random baseline gets the exact same number of evals.
            ordering_grid = {
                "expansion_budget_modifier": [self.budget_modifier],
                "time_budget_s": [-1],
                "exploration_c": [0.1, 0.25, 0.5, 1.0],
                "adjust_C_p": [True],
                "rollout_policy": ["random", "greedy"],
                "rollout_depth": [0, 2, 4],
                "max_indegree": [self.get_ordering_max_indegree()],
                # BIC rewards are large negative numbers; normalising to [0,1] makes
                # exploration_c interpretable as a fraction of the observed value range
                # regardless of graph size or regression type.
                "normalize_rewards": [True],
            }
            # Random baseline has no meaningful hyperparameters to optimise:
            # max_evals is always injected by SachsOrderingGroupEvalTask at eval time.
            hyperparam_grid_base = {
                OrderingMCTSAgent.algorithm_name: ordering_grid,
            }
        else:
            # uct_only / edge_uct: time-budget mode, sentinel expansion_budget_modifier.
            ordering_grid = {
                "expansion_budget_modifier": [1_000_000],
                "time_budget_s": [T],
                "exploration_c": [0.025, 0.05, 0.075, 0.1],
                "adjust_C_p": [True],
                "rollout_policy": ["random"],
                "rollout_depth": [0],
                "max_indegree": [self.get_ordering_max_indegree()],
            }
            hyperparam_grid_base = {
                UCTFullDepthAgent.algorithm_name: deepcopy(base_mcts),
                OrderingMCTSAgent.algorithm_name: ordering_grid,
            }

        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)
        return hyperparam_grids


def get_conditions_for_experiment(which, instance_name, cmd_args):
    if hasattr(cmd_args, "__dict__"):
        vars_dict = vars(cmd_args)
    else:
        vars_dict = cmd_args

    if which == "time_budget":
        cond = TimeBudgetExperimentConditions(instance_name, vars_dict)
    elif which == "sachs":
        cond = SachsExperimentConditions(instance_name, vars_dict)
    else:
        raise ValueError(f"experiment {which} not recognized!")
    return cond
