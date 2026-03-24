import pytest

from cdrl.agent.mcts.mcts_agent import (
    UCTDepth2Agent,
    UCTDepth4Agent,
    UCTDepth8Agent,
    UCTDepth16Agent,
    UCTDepth32Agent,
    UCTFullDepthAgent,
)
from cdrl.environment.graph_edge_env import DirectedGraphEdgeEnv, EnvPhase
from cdrl.io.file_paths import FilePaths
from cdrl.io.storage import EvaluationStorage
from cdrl.reward_functions.reward_continuous_vars import ContinuousVarsBICRewardFunction
from cdrl.state.instance_generators import HardcodedInstanceGenerator


@pytest.mark.parametrize(
    "agent_class",
    [
        UCTFullDepthAgent,
        UCTDepth2Agent,
        UCTDepth4Agent,
        UCTDepth8Agent,
        UCTDepth16Agent,
        UCTDepth32Agent,
    ],
)
def test_agent_runs(agent_class):
    inst_name = "sachs"
    score_type = "BIC_different_var"
    penalise_cyclic = True
    enforce_acyclic = True
    store_scores = True
    starting_graph_generation = "scratch"

    disc_inst = HardcodedInstanceGenerator.get_instance(
        instance_name=inst_name,
        normalize_data=True,
        starting_graph_generation=starting_graph_generation,
    )
    rfun = ContinuousVarsBICRewardFunction(
        disc_inst,
        penalise_cyclic=penalise_cyclic,
        score_type=score_type,
        store_scores=store_scores,
        reg_type=disc_inst.instance_metadata.reg_type,
    )

    initial_edge_budgets = {"construct": disc_inst.true_num_edges, "prune": 0}

    env = DirectedGraphEdgeEnv(
        disc_inst,
        rfun,
        initial_edge_budgets=initial_edge_budgets,
        enforce_acyclic=enforce_acyclic,
    )
    agent = agent_class(env)

    hyps = agent.get_default_hyperparameters()
    hyps["expansion_budget_modifier"] = 1

    opts = {
        "storage": EvaluationStorage(
            FilePaths("/experiment_data", "development", setup_directories=False)
        ),
        "model_identifier_prefix": "default",
        "log_progress": False,
        "log_timings": False,
        "random_seed": 42,
    }

    agent.setup(opts, hyps)
    agent.eval([disc_inst.start_state], EnvPhase.CONSTRUCT)
