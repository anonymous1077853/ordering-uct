import time
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from cdrl.agent.base_agent import Agent
from cdrl.environment.graph_edge_env import EnvPhase
from cdrl.agent.ordering.ordering_local_score import OrderingLocalScoreCache
from cdrl.agent.ordering.ordering_decoder import decode_ordering


class RandomOrderingBaseline(Agent):
    """
    Baseline that samples k random variable orderings and returns metrics
    for the best one found (after decoding each ordering and optimising for reward).

    k is set equal to the total number of orderings evaluated by the
    CD-Ordering-UCT agent under the same expansion_budget_modifier, ensuring
    a fair budget comparison:

        num_simulations_per_step = max(1, int(d * expansion_budget_modifier))
        k = d * num_simulations_per_step

    For each of the k orderings:
      1. Sample a uniformly random permutation of {0, ..., d-1}.
      2. Decode via Stage 1+2 (no pruning) using decode_ordering().
      3. Score with rfun.calculate_reward_single_graph().

    The best ordering is then decoded a final time with Stage 3 pruning applied,
    matching the reporting convention of OrderingMCTSAgent.

    Hyperparameters (set via setup()):
        expansion_budget_modifier : determines k (same parameter as ordering agent)
        max_indegree              : decoder Stage-2 max parents per node (default -1 = auto)
        coeff_threshold           : Stage-3 LR/QR pruning threshold (default 0.3)
        time_budget_s             : wall-clock time limit in seconds (-1 = disabled, use k)
    """

    algorithm_name = "random_ordering"
    is_deterministic = False
    is_trainable = False

    def __init__(self, environment) -> None:
        super().__init__(environment)

    # Agent interface

    def make_actions(self, t, **kwargs):
        raise ValueError(
            "RandomOrderingBaseline does not use the step-by-step edge MDP; "
            "call eval() directly."
        )

    def eval(self, g_list, phase):
        """
        Sample k random orderings and return the best decoded DAG.

        Args:
            g_list: List of initial DAGState objects (must have length 1).
            phase:  EnvPhase (only CONSTRUCT is meaningful here).

        Returns:
            (graphs, acts, rewards) where:
              graphs[0]  - DAGState of the best ordering found
              acts[0]    - list of variable indices (complete ordering)
              rewards[0] - rfun.calculate_reward_single_graph(graphs[0])
        """
        if len(g_list) > 1:
            raise ValueError("RandomOrderingBaseline supports only one graph at a time.")

        rfun = self.environment.reward_function
        X = rfun.inputdata
        d = rfun.d
        bic_penalty = rfun.bic_penalty
        score_type = rfun.score_type
        reg_type = self.environment.disc_instance.instance_metadata.reg_type

        expansion_budget_modifier = self.hyperparams.get("expansion_budget_modifier", 25)
        max_indegree = self.hyperparams.get("max_indegree", -1)
        coeff_threshold = self.hyperparams.get("coeff_threshold", 0.3)
        time_budget_s = self.hyperparams.get("time_budget_s", -1)

        max_evals_override = self.hyperparams.get("max_evals", None)
        if max_evals_override is not None:
            k = int(max_evals_override)
        else:
            num_simulations_per_step = max(1, int(d * expansion_budget_modifier))
            k = d * num_simulations_per_step

        rng = np.random.default_rng(self.random_seed)
        cache = OrderingLocalScoreCache()
        if reg_type == "LR":
            cache.precompute(X)

        best_reward: float = float("-inf")
        best_order: Optional[Tuple[int, ...]] = None

        start_time = time.time()
        variables = np.arange(d)
        with tqdm(
            total=k,
            colour="green",
            desc="Random-Ordering Baseline",
            disable=self.disable_tqdm,
        ) as pbar:
            for _ in range(k):
                if time_budget_s > 0 and (time.time() - start_time) >= time_budget_s:
                    break
                order = tuple(int(v) for v in rng.permutation(variables))
                dag = decode_ordering(
                    X, order, reg_type, bic_penalty, score_type,
                    max_indegree, coeff_threshold, cache=cache,
                    apply_pruning=False,
                )
                reward = float(rfun.calculate_reward_single_graph(dag, _skip_cycle_check=True))
                if reward > best_reward:
                    best_reward = reward
                    best_order = order
                pbar.update(1)

        # Final decode without Stage 3 pruning - Stage 3 is applied by the metrics
        # pipeline (prune_cam) for a fair apples-to-apples comparison with ordering UCT.
        best_dag = decode_ordering(
            X, best_order, reg_type, bic_penalty, score_type,
            max_indegree, coeff_threshold, cache=cache,
            apply_pruning=False,
        )
        best_reward = float(rfun.calculate_reward_single_graph(best_dag, _skip_cycle_check=True))

        return [best_dag], [list(best_order)], np.array([best_reward])

    def get_default_hyperparameters(self):
        return {
            "expansion_budget_modifier": 25,
            "max_indegree": -1,
            "coeff_threshold": 0.3,
            "time_budget_s": -1,
        }

    def finalize(self):
        pass
