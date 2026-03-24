"""
FilePaths: centralised management of experiment directory layout and filename conventions.

All paths for a given experiment (datasets, models, results, tasks, figures, logs) are
derived from a single `parent_dir` / `experiment_id` pair, making it straightforward to
relocate or reproduce an experiment by changing just those two values.
"""

import os
import subprocess
from copy import copy
from pathlib import Path


class FilePaths:
    """
    Manages the directory tree for a single experiment run.

    On construction, all required sub-directories are created under
    `parent_dir / experiment_id` (unless `setup_directories=False`).
    Constants define the sub-directory names so that the layout can be
    understood at a glance without reading the constructor.
    """
    DATE_FORMAT = "%Y-%m-%d-%H-%M-%S"
    DATASETS_DIR_NAME = 'datasets'
    SCORES_DIR_NAME = 'scores'

    MODELS_DIR_NAME = 'models'
    CHECKPOINTS_DIR_NAME = 'checkpoints'
    SUMMARIES_DIR_NAME = 'summaries'
    EVAL_HISTORIES_DIR_NAME = 'eval_histories'
    TEST_HISTORIES_DIR_NAME = 'test_histories'

    TIMINGS_DIR_NAME = 'timings'

    HYPEROPT_RESULTS_DIR_NAME = 'hyperopt_results'
    EVAL_RESULTS_DIR_NAME = 'eval_results'

    HYPEROPT_TASKS_DIR_NAME = 'tasks_hyperopt'
    EVAL_TASKS_DIR_NAME = 'tasks_eval'

    FIGURES_DIR_NAME = 'figures'
    TRAJECTORIES_DATA_DIR_NAME = 'trajectories_data'
    RLBIC_OUTPUT_DIR_NAME = 'rlbic_output'

    LOGS_DIR_NAME = 'logs'
    DEFAULT_MODEL_PREFIX = 'default'

    def __init__(self, parent_dir, experiment_id, setup_directories=True):
        """
        Initialises all directory paths for the experiment.

        `parent_dir` is the root storage directory (typically CD_EXPERIMENT_DATA_DIR).
        `experiment_id` is a unique string identifying this experiment (used as the
        sub-directory name). When `setup_directories=True`, all directories are created
        on disk and group read/write permissions are applied; set it to False to build
        path objects without touching the filesystem.
        """
        self.parent_dir = parent_dir
        self.experiment_id = experiment_id

        self.datasets_storage_dir = Path(self.parent_dir) / self.DATASETS_DIR_NAME
        self.datasets_storage_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = Path(self.parent_dir) / self.LOGS_DIR_NAME
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_dir = self.get_dir_for_experiment_id(experiment_id)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.experiment_dir / self.MODELS_DIR_NAME
        self.checkpoints_dir = self.models_dir / self.CHECKPOINTS_DIR_NAME
        self.summaries_dir = self.models_dir / self.SUMMARIES_DIR_NAME
        self.eval_histories_dir = self.models_dir / self.EVAL_HISTORIES_DIR_NAME
        self.test_histories_dir = self.models_dir / self.TEST_HISTORIES_DIR_NAME

        self.timings_dir = self.models_dir / self.TIMINGS_DIR_NAME

        self.hyperopt_results_dir = self.models_dir / self.HYPEROPT_RESULTS_DIR_NAME
        self.eval_results_dir = self.models_dir / self.EVAL_RESULTS_DIR_NAME

        self.hyperopt_tasks_dir = self.models_dir / self.HYPEROPT_TASKS_DIR_NAME
        self.eval_tasks_dir = self.models_dir / self.EVAL_TASKS_DIR_NAME

        self.figures_dir = self.experiment_dir / self.FIGURES_DIR_NAME

        self.trajectories_data_dir = self.experiment_dir / self.TRAJECTORIES_DATA_DIR_NAME
        self.rlbic_output_dir = self.experiment_dir / self.RLBIC_OUTPUT_DIR_NAME

        if setup_directories:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.summaries_dir.mkdir(parents=True, exist_ok=True)
            self.eval_histories_dir.mkdir(parents=True, exist_ok=True)
            self.test_histories_dir.mkdir(parents=True, exist_ok=True)

            self.timings_dir.mkdir(parents=True, exist_ok=True)

            self.hyperopt_results_dir.mkdir(parents=True, exist_ok=True)
            self.eval_results_dir.mkdir(parents=True, exist_ok=True)

            self.hyperopt_tasks_dir.mkdir(parents=True, exist_ok=True)
            self.eval_tasks_dir.mkdir(parents=True, exist_ok=True)

            self.figures_dir.mkdir(parents=True, exist_ok=True)
            self.trajectories_data_dir.mkdir(parents=True, exist_ok=True)
            self.rlbic_output_dir.mkdir(parents=True, exist_ok=True)

            self.set_group_permissions()


    def set_group_permissions(self):
        """Grants group read/write/execute access to the datasets and experiment directories.

        This allows multiple users sharing the same Unix group to read and write
        experiment outputs without ownership conflicts.
        """
        try:
            for dir in [self.datasets_storage_dir, self.experiment_dir]:
                abspath = str(dir.absolute())
                subprocess.run(["chmod", "-R", "g+rwx", abspath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def get_dir_for_experiment_id(self, experiment_id):
        """Returns the directory for a given experiment ID."""
        return Path(self.parent_dir) / f'{experiment_id}'

    def __str__(self):
        """Returns a dict-style string with every path attribute resolved to its absolute form."""
        asdict = self.__dict__
        target = copy(asdict)
        for param_name, corresp_path in asdict.items():
            target[param_name] = str(corresp_path.absolute())

        return str(target)

    def __repr__(self):
        return self.__str__()

    def construct_log_filepath(self):
        """Returns None - logging goes to stdout by default.

        Subclasses can override this to direct output to a file path.
        """
        return None

    @staticmethod
    def construct_task_filename(task_type, task_id):
        """Returns the corresponding task file associated with a particular task type and ID."""
        return f"{task_type}_{task_id}.obj"

    @staticmethod
    def construct_model_identifier_prefix(agent_name, obj_fun_name, network_generator_name, model_seed,  hyperparams_id, graph_id=None):
        """Returns the filename prefix used for hyperparameter tuning or evaluation result files."""
        model_identifier_prefix = f"{agent_name}-{obj_fun_name}-{network_generator_name}-{(graph_id + '-') if graph_id is not None else ''}" \
                                  f"{model_seed}-{hyperparams_id}"
        return model_identifier_prefix

    @staticmethod
    def construct_history_file_name(model_identifier_prefix):
        """Returns the filename for the history file."""
        return f"{model_identifier_prefix}_history.csv"

    @staticmethod
    def construct_timings_file_name(model_identifier_prefix):
        """Returns the filename for the timings file."""
        return f"{model_identifier_prefix}_timings.csv"

    @staticmethod
    def construct_best_validation_file_name(model_identifier_prefix):
        """Returns the filename for the validation file."""
        return f"{model_identifier_prefix}_best.csv"