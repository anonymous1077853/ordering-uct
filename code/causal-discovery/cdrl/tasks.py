import argparse
import json
import pickle
import time
from copy import deepcopy
import traceback
import sys
from datetime import datetime

from pathlib import Path
d = Path(__file__).resolve().parents[1]
sys.path.append(str(d.absolute()))

from cdrl.environment.graph_edge_env import DirectedGraphEdgeEnv, EnvPhase
from cdrl.evaluation.eval_utils import get_metrics_dict, extract_validation_perf_from_metrics_dict
from cdrl.io.file_paths import FilePaths
from cdrl.reward_functions.reward_continuous_vars import ContinuousVarsBICRewardFunction
from cdrl.state.instance_generators import HardcodedInstanceGenerator, SynthGPInstanceGenerator
from cdrl.utils.config_utils import get_logger_instance, date_format


class OptimizeHyperparamsTask(object):
    """
    Represents a unit of execution for the hyperparameter optimization (hyperopt) phase.

    Each task covers one (agent, objective function, network generator) triple over a
    chunk of hyperparameter combinations and model seeds.  Tasks are serialized to disk
    and executed independently inside the Docker container via tasks.py --experiment_part hyperopt.

    BTM is always disabled during hyperopt (btm=False) to keep each evaluation
    independent and avoid any implicit favouritism toward longer search runs.
    """
    def __init__(self,
                 task_id,
                 agent,
                 objective_function,
                 network_generator,
                 experiment_conditions,
                 storage,
                 parameter_keys,
                 search_space_chunk,
                 model_seeds_chunk,
                 train_kwargs=None,
                 eval_make_action_kwargs=None,
                 additional_opts=None
                 ):
        self.task_id = task_id
        self.agent = agent
        self.objective_function = objective_function
        self.network_generator = network_generator
        self.experiment_conditions = experiment_conditions
        self.storage = storage
        self.parameter_keys = parameter_keys
        self.search_space_chunk = search_space_chunk
        self.model_seeds_chunk = model_seeds_chunk
        self.train_kwargs = train_kwargs
        self.eval_make_action_kwargs = eval_make_action_kwargs
        self.additional_opts = additional_opts

    def run(self):
        log_filename = self.storage.file_paths.construct_log_filepath()
        logger = get_logger_instance(log_filename)
        total_runs = 0
        failed_runs = 0
        task_start = datetime.now()

        if self.network_generator == HardcodedInstanceGenerator:
            disc_inst = self.network_generator.get_instance(instance_name=self.experiment_conditions.instance_name, normalize_data=True, starting_graph_generation=self.experiment_conditions.starting_graph_generation)
            rfun = self.objective_function(disc_inst, reg_type=disc_inst.instance_metadata.reg_type, **self.experiment_conditions.experiment_params)


        for model_seed in self.model_seeds_chunk:
            exp_copy = deepcopy(self.experiment_conditions)

            if self.network_generator == SynthGPInstanceGenerator:
                disc_inst = self.network_generator.get_instance(instance_name=exp_copy.instance_name, gt=exp_copy.gt, n=exp_copy.n, p=exp_copy.p, e=exp_copy.e, what_vary=exp_copy.what_vary, model_seed=model_seed, normalize_data=True,
                                                                starting_graph_generation=exp_copy.starting_graph_generation)
                rfun = self.objective_function(disc_inst, reg_type=disc_inst.instance_metadata.reg_type, **exp_copy.experiment_params)


            for hyperparams_id, combination in self.search_space_chunk:
                hyperparams = {}

                for idx, param_value in enumerate(tuple(combination)):
                    param_key = self.parameter_keys[idx]
                    hyperparams[param_key] = param_value

                hyperparams['btm'] = False
                total_runs += 1

                model_identifier_prefix = self.storage.file_paths.construct_model_identifier_prefix(self.agent.algorithm_name,
                                                                                                    self.objective_function.name,
                                                                                                    self.network_generator.name,
                                                                                                    model_seed, hyperparams_id)

                env = DirectedGraphEdgeEnv(disc_inst, rfun, initial_edge_budgets=exp_copy.get_initial_edge_budgets(self.network_generator, disc_inst), **exp_copy.experiment_params)
                agent_instance = self.agent(env)

                run_options = {}
                run_options["random_seed"] = model_seed
                run_options["storage"] = self.storage
                run_options["file_paths"] = self.storage.file_paths

                run_options["log_progress"] = False
                run_options["disable_tqdm"] = True

                log_filename = self.storage.file_paths.construct_log_filepath()
                run_options["log_filename"] = log_filename
                run_options["model_identifier_prefix"] = model_identifier_prefix

                run_options['storage'] = self.storage
                run_options.update((self.additional_opts or {}))

                try:
                    agent_instance.setup(run_options, hyperparams)
                    if exp_copy.perform_construction:
                        construct_start = datetime.now()
                        construct_output = agent_instance.eval([disc_inst.start_state], EnvPhase.CONSTRUCT)
                        construct_end = datetime.now()
                        pruning_start_graph = construct_output[0][0]
                        duration_construct_s = (construct_end - construct_start).total_seconds()

                    else:
                        construct_output = None
                        pruning_start_graph = disc_inst.start_state
                        duration_construct_s = 0.

                    if exp_copy.perform_pruning:
                        prune_start = datetime.now()
                        prune_output = agent_instance.eval([pruning_start_graph], EnvPhase.PRUNE)
                        prune_end = datetime.now()
                        duration_prune_s = (prune_end - prune_start).total_seconds()
                    else:
                        prune_output = None
                        duration_prune_s = 0.

                    out_dict = get_metrics_dict(construct_output, prune_output, disc_inst, rfun, include_cam_pruning=True)
                    out_dict["hyperparams"] = hyperparams
                    out_dict["hyperparams_id"] = hyperparams_id

                    out_dict["duration_construct_s"] = duration_construct_s
                    out_dict["duration_prune_s"] = duration_prune_s

                    logger.info(
                        f"[hyperopt] agent={self.agent.algorithm_name} seed={model_seed} "
                        f"hyp_id={hyperparams_id} "
                        f"construct={duration_construct_s:.2f}s prune={duration_prune_s:.2f}s "
                        f"total={duration_construct_s + duration_prune_s:.2f}s"
                    )

                    self.storage.write_metrics_dict(model_identifier_prefix, out_dict, "hyperopt")

                    perf = extract_validation_perf_from_metrics_dict(out_dict)

                    self.storage.write_hyperopt_results(model_identifier_prefix, perf)
                    agent_instance.finalize()

                except BaseException:
                    failed_runs += 1
                    logger.warn("got exception while training & evaluating agent")
                    logger.warn(traceback.format_exc())
                    agent_instance.finalize()

        task_duration_s = (datetime.now() - task_start).total_seconds()
        return {"total_runs": total_runs, "failed_runs": failed_runs, "task_duration_s": task_duration_s}


class EvaluateTask(object):
    """
    Represents a unit of execution for the evaluation phase.

    Each task runs a single (agent, objective function, network generator, seed) combination
    using the best hyperparameters found during hyperopt.  BTM is enabled/disabled per agent
    as specified by experiment_conditions.btm_on_eval (which may be a bool or a dict keyed
    by algorithm_name).
    """
    def __init__(self,
                 task_id,
                 agent,
                 objective_function,
                 network_generator,
                 best_hyperparams,
                 best_hyperparams_id,
                 experiment_conditions,
                 storage,
                 model_seeds_chunk,
                 eval_make_action_kwargs=None,
                 additional_opts=None):
        self.task_id = task_id
        self.agent = agent
        self.objective_function = objective_function
        self.network_generator = network_generator

        self.best_hyperparams = best_hyperparams
        self.best_hyperparams_id = best_hyperparams_id

        self.experiment_conditions = experiment_conditions
        self.storage = storage
        self.model_seeds_chunk = model_seeds_chunk
        self.eval_make_action_kwargs = eval_make_action_kwargs
        self.additional_opts = additional_opts

    def run(self):
        log_filename = self.storage.file_paths.construct_log_filepath()
        logger = get_logger_instance(log_filename)
        total_runs = 0
        failed_runs = 0
        task_start = datetime.now()

        if self.network_generator == HardcodedInstanceGenerator:
            disc_inst = self.network_generator.get_instance(instance_name=self.experiment_conditions.instance_name, normalize_data=True, starting_graph_generation=self.experiment_conditions.starting_graph_generation)
            rfun = self.objective_function(disc_inst, reg_type=disc_inst.instance_metadata.reg_type, **self.experiment_conditions.experiment_params)


        for model_seed in self.model_seeds_chunk:
            total_runs += 1
            exp_copy = deepcopy(self.experiment_conditions)
            setting = (self.network_generator.name, self.objective_function.name, self.agent.algorithm_name)

            model_identifier_prefix = self.storage.file_paths.construct_model_identifier_prefix(self.agent.algorithm_name,
                                                                                                self.objective_function.name,
                                                                                                self.network_generator.name,
                                                                                                model_seed, self.best_hyperparams_id)
            if self.network_generator == SynthGPInstanceGenerator:
                disc_inst = self.network_generator.get_instance(instance_name=exp_copy.instance_name, gt=exp_copy.gt, n=exp_copy.n, p=exp_copy.p, e=exp_copy.e,  what_vary=exp_copy.what_vary, model_seed=model_seed, normalize_data=True,
                                                                starting_graph_generation=exp_copy.starting_graph_generation)
                rfun = self.objective_function(disc_inst, reg_type=disc_inst.instance_metadata.reg_type, **exp_copy.experiment_params)

            env = DirectedGraphEdgeEnv(disc_inst, rfun, initial_edge_budgets=exp_copy.get_initial_edge_budgets(self.network_generator, disc_inst), **exp_copy.experiment_params)
            agent_instance = self.agent(env)

            run_options = {}
            run_options["random_seed"] = model_seed
            run_options["file_paths"] = self.storage.file_paths
            run_options["log_progress"] = False
            run_options["disable_tqdm"] = True

            log_filename = self.storage.file_paths.construct_log_filepath()
            run_options["log_filename"] = log_filename
            run_options["model_identifier_prefix"] = model_identifier_prefix

            run_options['storage'] = self.storage


            run_options.update((self.additional_opts or {}))

            try:
                hyps_copy = deepcopy(self.best_hyperparams)
                btm_val = exp_copy.btm_on_eval
                if isinstance(btm_val, dict):
                    btm_val = btm_val.get(self.agent.algorithm_name, True)
                hyps_copy['btm'] = btm_val

                agent_instance.setup(run_options, hyps_copy)
                if exp_copy.perform_construction:
                    construct_start = datetime.now()
                    construct_output = agent_instance.eval([disc_inst.start_state], EnvPhase.CONSTRUCT)
                    construct_end = datetime.now()
                    pruning_start_graph = construct_output[0][0]
                    duration_construct_s = (construct_end - construct_start).total_seconds()

                else:
                    construct_output = None
                    pruning_start_graph = disc_inst.start_state
                    duration_construct_s = 0.

                if exp_copy.perform_pruning:
                    prune_start = datetime.now()
                    prune_output = agent_instance.eval([pruning_start_graph], EnvPhase.PRUNE)
                    prune_end = datetime.now()
                    duration_prune_s = (prune_end - prune_start).total_seconds()
                else:
                    prune_output = None
                    duration_prune_s = 0.

                out_dict = get_metrics_dict(construct_output, prune_output, disc_inst, rfun, include_cam_pruning=True)
                out_dict["hyperparams"] = hyps_copy
                out_dict["hyperparams_id"] = self.best_hyperparams_id

                out_dict["duration_construct_s"] = duration_construct_s
                out_dict["duration_prune_s"] = duration_prune_s

                logger.info(
                    f"[eval] agent={self.agent.algorithm_name} seed={model_seed} "
                    f"hyp_id={self.best_hyperparams_id} "
                    f"construct={duration_construct_s:.2f}s prune={duration_prune_s:.2f}s "
                    f"total={duration_construct_s + duration_prune_s:.2f}s"
                )

                self.storage.write_metrics_dict(model_identifier_prefix, out_dict, "eval")

                agent_instance.finalize()

            except BaseException:
                failed_runs += 1
                logger.warn("got exception while training & evaluating agent")
                logger.warn(traceback.format_exc())
                agent_instance.finalize()

        task_duration_s = (datetime.now() - task_start).total_seconds()
        return {"total_runs": total_runs, "failed_runs": failed_runs, "task_duration_s": task_duration_s}




class SachsOrderingGroupEvalTask:
    """
    Eval task that groups OrderingMCTSAgent + one or more baseline agents
    into a single sequential run per seed.

    Ordering UCT runs first (with its configured budget). Its num_simulations
    is then passed as max_evals to each baseline, giving an exact
    simulation-count-fair comparison per seed.
    """

    # Sentinel agent attribute so main() can read algorithm_name.
    class _AgentNameProxy:
        algorithm_name = "sachs_ordering_group"

    agent = _AgentNameProxy()

    def __init__(self, task_id, ordering_agent, baseline_agents,
                 objective_function, network_generator,
                 ordering_hyperparams, ordering_hyperparams_id,
                 baseline_hyperparams_list, baseline_hyperparams_ids,
                 experiment_conditions, storage, model_seeds_chunk,
                 additional_opts=None):
        self.task_id = task_id
        self.ordering_agent = ordering_agent
        self.baseline_agents = baseline_agents
        self.objective_function = objective_function
        self.network_generator = network_generator
        self.ordering_hyperparams = ordering_hyperparams
        self.ordering_hyperparams_id = ordering_hyperparams_id
        self.baseline_hyperparams_list = baseline_hyperparams_list
        self.baseline_hyperparams_ids = baseline_hyperparams_ids
        self.experiment_conditions = experiment_conditions
        self.storage = storage
        self.model_seeds_chunk = model_seeds_chunk
        self.additional_opts = additional_opts

    def _make_run_options(self, model_seed, model_identifier_prefix):
        log_filename = self.storage.file_paths.construct_log_filepath()
        return {
            "random_seed": model_seed,
            "storage": self.storage,
            "file_paths": self.storage.file_paths,
            "log_filename": log_filename,
            "model_identifier_prefix": model_identifier_prefix,
            "log_progress": False,
            "disable_tqdm": True,
        }

    def run(self):
        log_filename = self.storage.file_paths.construct_log_filepath()
        logger = get_logger_instance(log_filename)
        total_runs = 0
        failed_runs = 0
        task_start = datetime.now()

        # Sachs is always a hardcoded instance (fixed dataset).
        disc_inst = self.network_generator.get_instance(
            instance_name=self.experiment_conditions.instance_name,
            normalize_data=True,
            starting_graph_generation=self.experiment_conditions.starting_graph_generation,
        )
        rfun = self.objective_function(
            disc_inst,
            reg_type=disc_inst.instance_metadata.reg_type,
            **self.experiment_conditions.experiment_params,
        )

        for model_seed in self.model_seeds_chunk:
            exp_copy = deepcopy(self.experiment_conditions)
            total_runs += 1

            try:
                # Step 1: Run OrderingMCTSAgent with time_budget_s.
                ordering_prefix = self.storage.file_paths.construct_model_identifier_prefix(
                    self.ordering_agent.algorithm_name,
                    self.objective_function.name,
                    self.network_generator.name,
                    model_seed,
                    self.ordering_hyperparams_id,
                )
                run_options = self._make_run_options(model_seed, ordering_prefix)
                run_options.update(self.additional_opts or {})

                ordering_env = DirectedGraphEdgeEnv(
                    disc_inst, rfun,
                    initial_edge_budgets=exp_copy.get_initial_edge_budgets(
                        self.network_generator, disc_inst),
                    **exp_copy.experiment_params,
                )
                ordering_instance = self.ordering_agent(ordering_env)

                ordering_hyps = deepcopy(self.ordering_hyperparams)
                btm_val = exp_copy.btm_on_eval
                if isinstance(btm_val, dict):
                    btm_val = btm_val.get(self.ordering_agent.algorithm_name, True)
                ordering_hyps["btm"] = btm_val

                ordering_instance.setup(run_options, ordering_hyps)

                construct_start = datetime.now()
                construct_output = ordering_instance.eval(
                    [disc_inst.start_state], EnvPhase.CONSTRUCT)
                construct_end = datetime.now()
                duration_construct_s = (construct_end - construct_start).total_seconds()

                num_simulations = ordering_instance.num_simulations

                out_dict = get_metrics_dict(
                    construct_output, None, disc_inst, rfun, include_cam_pruning=True)
                out_dict["hyperparams"] = ordering_hyps
                out_dict["hyperparams_id"] = self.ordering_hyperparams_id
                out_dict["duration_construct_s"] = duration_construct_s
                out_dict["duration_prune_s"] = 0.0
                out_dict["num_simulations"] = num_simulations
                self.storage.write_metrics_dict(ordering_prefix, out_dict, "eval")

                logger.info(
                    f"[eval] agent={self.ordering_agent.algorithm_name} seed={model_seed} "
                    f"construct={duration_construct_s:.2f}s "
                    f"num_simulations={num_simulations}"
                )
                ordering_instance.finalize()

                # Step 2: Run each baseline with max_evals = num_simulations.
                for baseline_agent, baseline_hyps, baseline_hyps_id in zip(
                    self.baseline_agents,
                    self.baseline_hyperparams_list,
                    self.baseline_hyperparams_ids,
                ):
                    baseline_prefix = self.storage.file_paths.construct_model_identifier_prefix(
                        baseline_agent.algorithm_name,
                        self.objective_function.name,
                        self.network_generator.name,
                        model_seed,
                        baseline_hyps_id,
                    )
                    b_run_options = self._make_run_options(model_seed, baseline_prefix)
                    b_run_options.update(self.additional_opts or {})

                    baseline_env = DirectedGraphEdgeEnv(
                        disc_inst, rfun,
                        initial_edge_budgets=exp_copy.get_initial_edge_budgets(
                            self.network_generator, disc_inst),
                        **exp_copy.experiment_params,
                    )
                    baseline_instance = baseline_agent(baseline_env)

                    b_hyps = deepcopy(baseline_hyps)
                    b_hyps["max_evals"] = num_simulations
                    b_hyps["max_indegree"] = ordering_hyps["max_indegree"]
                    b_hyps["time_budget_s"] = -1

                    baseline_instance.setup(b_run_options, b_hyps)

                    b_construct_start = datetime.now()
                    b_construct_output = baseline_instance.eval(
                        [disc_inst.start_state], EnvPhase.CONSTRUCT)
                    b_construct_end = datetime.now()
                    b_duration_s = (b_construct_end - b_construct_start).total_seconds()

                    b_out_dict = get_metrics_dict(
                        b_construct_output, None, disc_inst, rfun, include_cam_pruning=True)
                    b_out_dict["hyperparams"] = b_hyps
                    b_out_dict["hyperparams_id"] = baseline_hyps_id
                    b_out_dict["duration_construct_s"] = b_duration_s
                    b_out_dict["duration_prune_s"] = 0.0
                    self.storage.write_metrics_dict(baseline_prefix, b_out_dict, "eval")

                    logger.info(
                        f"[eval] agent={baseline_agent.algorithm_name} seed={model_seed} "
                        f"construct={b_duration_s:.2f}s max_evals={b_hyps.get('max_evals', 'N/A')}"
                    )
                    baseline_instance.finalize()

            except BaseException:
                failed_runs += 1
                logger.warn("got exception in SachsOrderingGroupEvalTask")
                logger.warn(traceback.format_exc())

        task_duration_s = (datetime.now() - task_start).total_seconds()
        return {"total_runs": total_runs, "failed_runs": failed_runs,
                "task_duration_s": task_duration_s}


def main():
    parser = argparse.ArgumentParser(description="Run a given task.")
    parser.add_argument("--experiment_part", required=True, type=str,
                        help="Whether to setup hyperparameter optimisation or evaluation.",
                        choices=["hyperopt", "eval"])

    parser.add_argument("--parent_dir", type=str, help="Root path for storing experiment data.")
    parser.add_argument("--experiment_id", required=True, help="experiment id to use")

    parser.add_argument("--task_id", type=str, required=True, help="Task id to run. Must have already been generated.")
    parser.set_defaults(parent_dir="/experiment_data")

    args = parser.parse_args()

    file_paths = FilePaths(args.parent_dir, args.experiment_id, setup_directories=False)
    task_storage_dir = file_paths.hyperopt_tasks_dir if args.experiment_part == "hyperopt" else file_paths.eval_tasks_dir
    task_file = task_storage_dir / FilePaths.construct_task_filename(args.experiment_part, args.task_id)
    with open(task_file, 'rb') as fh:
        task = pickle.load(fh)

    summary = task.run()
    agent_name = getattr(task.agent, "algorithm_name", str(task.agent))
    total_runs = summary.get("total_runs", 0) if isinstance(summary, dict) else 0
    failed_runs = summary.get("failed_runs", 0) if isinstance(summary, dict) else 0
    print(
        f"[task:{args.experiment_part}:{args.task_id}] "
        f"agent={agent_name} runs={total_runs} failed={failed_runs} "
        f"duration={summary.get('task_duration_s', 0.):.1f}s"
    )



if __name__ == "__main__":
    main()


























































