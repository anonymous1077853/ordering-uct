import math
import time
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from cdrl.agent.base_agent import Agent
from cdrl.environment.graph_edge_env import EnvPhase
from cdrl.state.dag_state import DAGState
from cdrl.agent.ordering.ordering_state import OrderingState
from cdrl.agent.ordering.ordering_node import OrderingNode
from cdrl.agent.ordering.ordering_local_score import OrderingLocalScoreCache
from cdrl.agent.ordering.ordering_decoder import decode_ordering, _decode_stage2_one
from cdrl.agent.ordering.ordering_rollout import rollout_random, rollout_greedy


class OrderingMCTSAgent(Agent):
    """
    MCTS agent that searches over variable orderings (permutations of {0,...,d-1})
    and decodes the best ordering found into a DAG.

    For each of the d variable positions in the ordering, a fresh UCT tree is
    built using ``int(d * expansion_budget_modifier)`` MCTS simulations.  After
    all simulations, the variable with the highest visit count at the root is
    committed (robust-child rule).  When BTM is enabled the best complete
    ordering seen across every simulation across every step is returned instead.

    Each MCTS simulation:
        Selection   - UCT on raw Q values
        Expansion   - one untried child created per simulation
        Rollout     - random or greedy completion -> decode (3-stage) -> rfun score
        Backprop    - N, W updated along the path from root to expanded node

    Integration:
        Receives a DirectedGraphEdgeEnv (created by tasks.py / run_causal_discovery.py)
        and extracts data / reward-function from it.  ``eval()`` is overridden
        completely; ``make_actions()`` raises an error as it is never called.

    Hyperparameters (set via setup()):
        expansion_budget_modifier : simulations per step = int(d * modifier)
                                    (same parameter name as edge-MCTS for budget parity)
        exploration_c             : UCT exploration constant (replaces C_p)
        adjust_C_p               : if True, rescale exploration_c by |root.Q|
                                   after each ordering step (edge-CD-UCT parity)
        rollout_policy            : "random" or "greedy"
        rollout_depth             : 0 = full rollout; k > 0 = truncated
        max_indegree              : decoder Stage-2 max parents per node (default 3)
        coeff_threshold           : Stage-3 LR/QR pruning threshold (default 0.3)
        btm                       : track best trajectory (injected by tasks.py)
    """

    algorithm_name = "ordering_uct"
    is_deterministic = False
    is_trainable = False

    def __init__(self, environment) -> None:
        super().__init__(environment)
        self.num_simulations = 0

    # Agent interface

    def make_actions(self, t, **kwargs):
        raise ValueError(
            "OrderingMCTSAgent does not use the step-by-step edge MDP; "
            "call eval() directly."
        )

    def eval(self, g_list, phase):
        """
        Run the ordering MCTS search.

        Args:
            g_list: List of initial DAGState objects (must have length 1).
            phase:  EnvPhase (CONSTRUCT or PRUNE - only CONSTRUCT is meaningful
                    here; the ordering search always runs the full discovery).

        Returns:
            (graphs, acts, rewards) where:
              graphs[0]  - DAGState of the best ordering found (BTM when enabled)
              acts[0]    - list of variable indices (complete ordering)
              rewards[0] - rfun.calculate_reward_single_graph(graphs[0])
        """
        if len(g_list) > 1:
            raise ValueError("OrderingMCTSAgent supports only one graph at a time.")

        rfun = self.environment.reward_function
        X = rfun.inputdata          # normalised data (n, d)
        d = rfun.d
        bic_penalty = rfun.bic_penalty   # log(n)/n
        score_type = rfun.score_type
        reg_type = self.environment.disc_instance.instance_metadata.reg_type

        expansion_budget_modifier = self.hyperparams.get("expansion_budget_modifier", 25)
        exploration_c = self.hyperparams.get("exploration_c", 1.4)
        rollout_policy = self.hyperparams.get("rollout_policy", "random")
        rollout_depth = self.hyperparams.get("rollout_depth", 0)
        max_indegree = self.hyperparams.get("max_indegree", -1)
        coeff_threshold = self.hyperparams.get("coeff_threshold", 0.3)
        btm = self.hyperparams.get("btm", True)
        time_budget_s = self.hyperparams.get("time_budget_s", -1)
        adjust_C_p = self.hyperparams.get("adjust_C_p", True)
        normalize_rewards = self.hyperparams.get("normalize_rewards", False)

        num_simulations_per_step = max(1, int(d * expansion_budget_modifier))

        rng = np.random.default_rng(self.random_seed)

        best_dag, best_order, best_reward, sim_count = self._run_search(
            X=X,
            d=d,
            reg_type=reg_type,
            bic_penalty=bic_penalty,
            score_type=score_type,
            rfun=rfun,
            num_simulations_per_step=num_simulations_per_step,
            exploration_c=exploration_c,
            rollout_policy=rollout_policy,
            rollout_depth=rollout_depth,
            max_indegree=max_indegree,
            coeff_threshold=coeff_threshold,
            btm=btm,
            rng=rng,
            time_budget_s=time_budget_s,
            adjust_C_p=adjust_C_p,
            normalize_rewards=normalize_rewards,
        )
        self.num_simulations = sim_count

        return [best_dag], [list(best_order)], np.array([best_reward])

    def get_default_hyperparameters(self):
        return {
            "expansion_budget_modifier": 25,
            "exploration_c": 1.4,
            "rollout_policy": "random",
            "rollout_depth": 0,
            "max_indegree": -1,
            "coeff_threshold": 0.3,
            "btm": True,
            "time_budget_s": -1,
            "adjust_C_p": True,
            "normalize_rewards": False,
        }

    def finalize(self):
        pass  # no resources to release

    # Core MCTS loop

    def _run_search(
        self,
        X: np.ndarray,
        d: int,
        reg_type: str,
        bic_penalty: float,
        score_type: str,
        rfun,
        num_simulations_per_step: int,
        exploration_c: float,
        rollout_policy: str,
        rollout_depth: int,
        max_indegree: int,
        coeff_threshold: float,
        btm: bool,
        rng: np.random.Generator,
        time_budget_s: float = -1,
        adjust_C_p: bool = True,
        normalize_rewards: bool = False,
    ) -> Tuple[DAGState, Tuple[int, ...], float]:
        """
        Main ordering MCTS loop.

        For each of the d positions in the ordering:
          1. Build a fresh UCT tree rooted at the current prefix.
          2. Run num_simulations_per_step MCTS iterations.
          3. Commit the variable with the highest visit count (robust child).
          4. Track the globally best complete ordering (BTM).

        Returns:
            (best_dag, best_order, best_reward)
        """
        # Shared RSS cache across all simulations for the entire eval() call.
        cache = OrderingLocalScoreCache()
        # Precompute the augmented Gram matrix for LR so that every cache miss
        # uses an O(k^2) slice + O(k^3) solve instead of O(n*k^2) matrix multiply.
        if reg_type == "LR":
            cache.precompute(X)

        # BTM (best trajectory memorisation) state
        best_reward: float = float("-inf")
        best_order: Optional[Tuple[int, ...]] = None
        sim_count: int = 0

        # Grow the committed ordering one variable at a time.
        current_prefix: Tuple[int, ...] = ()

        # Reward bounds tracked across all steps. Only consumed by _uct_value when
        # normalize_rewards=True; tracking them unconditionally is cheap and keeps
        # the warmup rollout below consistent regardless of the normalisation setting.
        reward_min = float("inf")
        reward_max = float("-inf")

        start_time = time.time()
        current_exploration_c = exploration_c

        # Allocate time proportional to available actions at each step.
        # Step k has (d-k) available actions; step d-1 (avail==1) is always skipped.
        # Early steps get more time (more choices to evaluate), late steps less.
        if time_budget_s > 0 and d > 1:
            weights = [d - k for k in range(d - 1)]   # [d, d-1, ..., 2]
            total_weight = sum(weights)                 # = d*(d+1)/2 - 1
            per_step_time_budgets = [time_budget_s * w / total_weight for w in weights]
        else:
            per_step_time_budgets = None

        # Warm-up rollout: run one complete rollout before the main loop to seed the
        # BTM tracker with an initial solution and to initialise reward_min/reward_max.
        # The bounds are needed by _uct_value when normalize_rewards=True; when
        # normalize_rewards=False they go unused but the rollout still gives BTM a
        # non-trivial starting point, so the warmup is always worthwhile.
        if adjust_C_p:
            _warmup_state = OrderingState(order=(), d=d)
            warmup_rollout = rollout_random if rollout_policy == "random" else rollout_greedy
            _q0, _order0, _ = warmup_rollout(
                X=X,
                state=_warmup_state,
                rng=rng,
                reg_type=reg_type,
                bic_penalty=bic_penalty,
                score_type=score_type,
                max_indegree=max_indegree,
                coeff_threshold=coeff_threshold,
                rfun=rfun,
                cache=cache,
                rollout_depth=rollout_depth,
            )
            if _q0 < reward_min:
                reward_min = _q0
            if _q0 > reward_max:
                reward_max = _q0
            if btm and _order0 is not None and len(_order0) == d and _q0 > best_reward:
                best_reward = _q0
                best_order = _order0

        with tqdm(total=d, colour="blue", desc="CD-Ordering-UCT", disable=self.disable_tqdm) as pbar:
            for step in range(d):
                # Safety net: stop before starting a new step if total budget exhausted.
                if time_budget_s > 0 and (time.time() - start_time) >= time_budget_s:
                    break

                current_state = OrderingState(order=current_prefix, d=d)
                if current_state.is_terminal():
                    break

                avail = current_state.available_actions()
                if len(avail) == 1:
                    # Only one variable left - commit without searching.
                    chosen = avail[0]
                    current_prefix = current_prefix + (chosen,)
                    # Precompute Stage 2 for the newly committed variable so that
                    # subsequent simulations can skip it via cache.prefix_parent_sets.
                    candidates = list(current_prefix[:-1])
                    cache.prefix_parent_sets[chosen] = _decode_stage2_one(
                        chosen, candidates, cache, X, reg_type, bic_penalty,
                        max_indegree, d, "BIC_different_var",
                    )
                    pbar.update(1)
                    continue

                root = OrderingNode(current_state)
                step_start_time = time.time()

                for _ in range(num_simulations_per_step):
                    # Per-step time check: stop this step's simulations when its slice is used up.
                    step_budget = per_step_time_budgets[step] if per_step_time_budgets is not None else -1
                    if step_budget > 0 and (time.time() - step_start_time) >= step_budget:
                        break

                    G, sim_order, sim_dag = self._simulate(
                        root=root,
                        X=X,
                        d=d,
                        reg_type=reg_type,
                        bic_penalty=bic_penalty,
                        score_type=score_type,
                        max_indegree=max_indegree,
                        coeff_threshold=coeff_threshold,
                        rfun=rfun,
                        rng=rng,
                        rollout_policy=rollout_policy,
                        rollout_depth=rollout_depth,
                        exploration_c=current_exploration_c,
                        reward_min=reward_min,
                        reward_max=reward_max,
                        cache=cache,
                        normalize_rewards=normalize_rewards,
                    )
                    sim_count += 1

                    # Keep running bounds (legacy bookkeeping; no longer used in UCT value).
                    if G < reward_min:
                        reward_min = G
                    if G > reward_max:
                        reward_max = G

                    # BTM: track the best complete ordering seen so far
                    if btm and sim_order is not None and len(sim_order) == d:
                        if G > best_reward:
                            best_reward = G
                            best_order = sim_order

                if adjust_C_p:
                    # Reward normalisation in _uct_value (using reward_min/max) keeps
                    # exploration_c scale-invariant; no per-step magnitude rescaling needed.
                    pass

                # Robust-child action selection: highest visit count
                if not root.children:
                    break
                chosen = max(root.children.items(), key=lambda kv: kv[1].N)[0]
                current_prefix = current_prefix + (chosen,)
                # Precompute Stage 2 for the newly committed variable so that
                # subsequent simulations can skip it via cache.prefix_parent_sets.
                candidates = list(current_prefix[:-1])
                cache.prefix_parent_sets[chosen] = _decode_stage2_one(
                    chosen, candidates, cache, X, reg_type, bic_penalty,
                    max_indegree, d, "BIC_different_var",
                )
                pbar.update(1)

        # Fallback: if BTM never recorded a complete ordering (e.g. btm=False),
        # use the committed prefix (possibly completed with random remaining).
        if best_order is None:
            committed = list(current_prefix)
            if len(committed) < d:
                remaining = [v for v in range(d) if v not in set(committed)]
                rng.shuffle(remaining)
                committed.extend(remaining)
            best_order = tuple(committed)

        # Final decode without Stage 3 pruning - results["construct"] reports Stage 1+2
        # metrics to match edge UCT's convention.  Stage 3 pruning is applied separately
        # by eval_utils.get_metrics_dict (prune_cam), giving both agents an apples-to-apples
        # "prune_cam" column for comparison.
        best_dag = decode_ordering(
            X, best_order, reg_type, bic_penalty, score_type,
            max_indegree, coeff_threshold, cache=cache,
            apply_pruning=False,
        )
        best_reward = float(rfun.calculate_reward_single_graph(best_dag, _skip_cycle_check=True))

        return best_dag, best_order, best_reward, sim_count

    # Single MCTS simulation

    def _simulate(
        self,
        root: OrderingNode,
        X: np.ndarray,
        d: int,
        reg_type: str,
        bic_penalty: float,
        score_type: str,
        max_indegree: int,
        coeff_threshold: float,
        rfun,
        rng: np.random.Generator,
        rollout_policy: str,
        rollout_depth: int,
        exploration_c: float,
        reward_min: float,
        reward_max: float,
        cache: OrderingLocalScoreCache,
        normalize_rewards: bool = False,
    ) -> Tuple[float, Optional[Tuple[int, ...]], Optional[DAGState]]:
        """
        One MCTS simulation: Selection -> Expansion -> Rollout -> Backpropagation.

        Returns:
            (reward, completed_order, dag) where completed_order and dag come
            from the rollout terminal and are used for BTM tracking.
        """
        path: List[OrderingNode] = [root]
        node = root

        # === Phase 1+2: Selection and Expansion ===
        while True:
            if node.state.is_terminal():
                # Reached a terminal node during selection - evaluate directly.
                dag = decode_ordering(
                    X, node.state.order, reg_type, bic_penalty, score_type,
                    max_indegree, coeff_threshold, cache=cache,
                    apply_pruning=False,
                )
                reward = float(rfun.calculate_reward_single_graph(dag, _skip_cycle_check=True))
                for n in path:
                    n.update(reward)
                return reward, node.state.order, dag

            avail = node.state.available_actions()
            untried = [a for a in avail if a not in node.children]

            if untried:
                # Expansion: create one new child for a randomly chosen untried action.
                a = int(rng.choice(untried))
                child_state = OrderingState(order=node.state.order + (a,), d=d)
                child = OrderingNode(child_state)
                node.children[a] = child
                node = child
                path.append(node)
                break
            else:
                # All children tried - UCT selection.
                node = self._uct_select(node, exploration_c, reward_min, reward_max, normalize_rewards)
                path.append(node)

        # === Phase 3: Rollout ===
        if node.state.is_terminal():
            dag = decode_ordering(
                X, node.state.order, reg_type, bic_penalty, score_type,
                max_indegree, coeff_threshold, cache=cache,
                apply_pruning=False,
            )
            reward = float(rfun.calculate_reward_single_graph(dag, _skip_cycle_check=True))
            completed_order = node.state.order
        else:
            if rollout_policy == "random":
                reward, completed_order, dag = rollout_random(
                    X=X,
                    state=node.state,
                    rng=rng,
                    reg_type=reg_type,
                    bic_penalty=bic_penalty,
                    score_type=score_type,
                    max_indegree=max_indegree,
                    coeff_threshold=coeff_threshold,
                    rfun=rfun,
                    cache=cache,
                    rollout_depth=rollout_depth,
                )
            elif rollout_policy == "greedy":
                reward, completed_order, dag = rollout_greedy(
                    X=X,
                    state=node.state,
                    rng=rng,
                    reg_type=reg_type,
                    bic_penalty=bic_penalty,
                    score_type=score_type,
                    max_indegree=max_indegree,
                    coeff_threshold=coeff_threshold,
                    rfun=rfun,
                    cache=cache,
                    rollout_depth=rollout_depth,
                )
            else:
                raise ValueError(f"Unknown rollout_policy: {rollout_policy!r}")

        # === Phase 4: Backpropagation ===
        for n in path:
            n.update(reward)

        return reward, completed_order, dag

    # UCT helpers

    @staticmethod
    def _uct_value(
        parent: OrderingNode,
        child: OrderingNode,
        exploration_c: float,
        reward_min: float,
        reward_max: float,
        normalize_rewards: bool = False,
    ) -> float:
        """
        UCT score. Unvisited children return +inf to force exploration first.

        When normalize_rewards=True, Q is min-max normalised to [0, 1] using running
        reward bounds so that exploration_c is scale-invariant regardless of reward
        magnitude (e.g. BIC vs GPR). Falls back to raw Q when bounds are degenerate.
        When normalize_rewards=False (default), raw Q is used - matching edge UCT.
        """
        if child.N == 0:
            return float("inf")
        q = child.Q
        if normalize_rewards:
            r_range = reward_max - reward_min
            if r_range > 1e-8:
                q = (q - reward_min) / r_range
        parent_visits = max(parent.N, 1)
        return q + exploration_c * math.sqrt((2.0 * math.log(parent_visits)) / child.N)

    @staticmethod
    def _uct_select(
        node: OrderingNode,
        exploration_c: float,
        reward_min: float,
        reward_max: float,
        normalize_rewards: bool = False,
    ) -> OrderingNode:
        """Select the child with the maximum UCT score."""
        best_val = float("-inf")
        best_child = None
        for child in node.children.values():
            val = OrderingMCTSAgent._uct_value(
                node, child, exploration_c, reward_min, reward_max, normalize_rewards
            )
            if val > best_val:
                best_val = val
                best_child = child
        return best_child
