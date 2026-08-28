"""Utilities supporting Optuna-based hyperparameter optimization."""

import csv
import json
import logging
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from equinet.args import HyperoptArgs
from equinet.constants import HYPEROPT_JOURNAL_FILE_NAME
from equinet.utils import makedirs


class SearchParameter:
    """
    Base class for a single entry of the hyperparameter search space.

    Each search parameter knows how to sample itself from an Optuna trial, which Optuna
    distributions back it, and how to encode an already known value back into Optuna
    parameters so that manually trained models can be added to a study as completed trials.
    """

    def suggest(self, name: str, trial: optuna.Trial) -> Any:
        """Samples a value for this parameter from an Optuna trial."""
        raise NotImplementedError

    def distributions(self, name: str) -> Dict[str, BaseDistribution]:
        """Returns the Optuna distributions backing this parameter, keyed by Optuna parameter name."""
        raise NotImplementedError

    def encode(self, name: str, value: Any) -> Dict[str, Any]:
        """Converts a known value of this parameter into the Optuna parameters that would produce it."""
        return {name: value}


class FloatParameter(SearchParameter):
    """A continuous search parameter, optionally discretized with a step or sampled in log space."""

    def __init__(self, low: float, high: float, step: float = None, log: bool = False):
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def suggest(self, name: str, trial: optuna.Trial) -> float:
        return trial.suggest_float(name, self.low, self.high, step=self.step, log=self.log)

    def distributions(self, name: str) -> Dict[str, BaseDistribution]:
        return {name: FloatDistribution(self.low, self.high, step=self.step, log=self.log)}


class IntParameter(SearchParameter):
    """An integer search parameter with a fixed step between allowed values."""

    def __init__(self, low: int, high: int, step: int = 1):
        self.low = low
        self.high = high
        self.step = step

    def suggest(self, name: str, trial: optuna.Trial) -> int:
        return trial.suggest_int(name, self.low, self.high, step=self.step)

    def distributions(self, name: str) -> Dict[str, BaseDistribution]:
        return {name: IntDistribution(self.low, self.high, step=self.step)}

    def encode(self, name: str, value: Any) -> Dict[str, Any]:
        return {name: int(value)}


class CategoricalParameter(SearchParameter):
    """A search parameter selected from a fixed list of options."""

    def __init__(self, options: Sequence[Any]):
        self.options = list(options)

    def suggest(self, name: str, trial: optuna.Trial) -> Any:
        return trial.suggest_categorical(name, self.options)

    def distributions(self, name: str) -> Dict[str, BaseDistribution]:
        return {name: CategoricalDistribution(self.options)}


class ZeroOrLogUniformParameter(SearchParameter):
    """
    A search parameter that is either exactly zero or drawn log-uniformly from a positive range.

    This is a conditional search space: the magnitude is only sampled when the nonzero branch
    is taken, so trials on the zero branch never record a magnitude parameter.
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def suggest(self, name: str, trial: optuna.Trial) -> float:
        if trial.suggest_categorical(f"{name}_nonzero", [False, True]):
            return trial.suggest_float(f"{name}_magnitude", self.low, self.high, log=True)
        return 0.0

    def distributions(self, name: str) -> Dict[str, BaseDistribution]:
        return {
            f"{name}_nonzero": CategoricalDistribution([False, True]),
            f"{name}_magnitude": FloatDistribution(self.low, self.high, log=True),
        }

    def encode(self, name: str, value: Any) -> Dict[str, Any]:
        if value == 0:
            return {f"{name}_nonzero": False}
        return {f"{name}_nonzero": True, f"{name}_magnitude": value}


def build_search_space(
    search_parameters: List[str], train_epochs: int = None
) -> Dict[str, SearchParameter]:
    """
    Builds the parameter space to be searched with Optuna trials.

    :param search_parameters: A list of parameters to be included in the search space.
    :param train_epochs: The total number of epochs to be used in training.
    :return: A dictionary keyed by the parameter names of :class:`SearchParameter` objects.
    """
    available_spaces = {
        "activation": CategoricalParameter(
            ["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU", "SiLU", "softplus"]
        ),
        "vp": CategoricalParameter([None, "antoine", "simplified"]),
        "vle": CategoricalParameter(["activity", "basic", "nrtl", "nrtl-wohl", "wohl"]),
        "wohl_order": IntParameter(low=3, high=5),
        "self_activity_correction": CategoricalParameter([True, False]),
        "self_activity_lambda": ZeroOrLogUniformParameter(low=1e-5, high=1e-1),
        "fugacity_balance": CategoricalParameter([True, False]),
        "aggregation": CategoricalParameter(["mean", "sum", "norm"]),
        "aggregation_norm": IntParameter(low=1, high=200),
        "batch_size": IntParameter(low=5, high=200, step=5),
        "depth": IntParameter(low=2, high=6),
        "dropout": FloatParameter(low=0.0, high=0.4, step=0.05),
        "ffn_hidden_size": IntParameter(low=300, high=2400, step=100),
        "ffn_num_layers": IntParameter(low=2, high=6),
        "final_lr_ratio": FloatParameter(low=1e-4, high=1.0, log=True),
        "hidden_size": IntParameter(low=300, high=2400, step=100),
        "init_lr_ratio": FloatParameter(low=1e-4, high=1.0, log=True),
        "linked_hidden_size": IntParameter(low=300, high=2400, step=100),
        "max_lr": FloatParameter(low=1e-6, high=1e-2, log=True),
        "weight_decay": FloatParameter(low=1e-6, high=1e-1, log=True),
    }  # TODO add any new parameters here
    if train_epochs is not None:
        available_spaces["warmup_epochs"] = IntParameter(low=1, high=train_epochs // 2)

    space = {}
    for key in search_parameters:
        space[key] = available_spaces[key]

    return space


def suggest_hyperparameters(
    trial: optuna.Trial, space: Dict[str, SearchParameter]
) -> Dict[str, Any]:
    """
    Samples one set of hyperparameters from the search space.

    :param trial: The Optuna trial to sample from.
    :param space: The search space, as returned by :func:`build_search_space`.
    :return: A dictionary keyed by the parameter names of the sampled values.
    """
    return {key: parameter.suggest(key, trial) for key, parameter in space.items()}


def build_storage(dir_path: str) -> JournalStorage:
    """
    Builds the file-backed Optuna storage used to share trials between instances.

    Optuna's journal storage appends each study operation to a single log file guarded by a
    lock file, which allows any number of independent hyperparameter optimization instances
    to contribute to and read from the same study as long as they share this directory. The
    open-based lock is used because it is the variant that behaves correctly on NFS.

    :param dir_path: Path to the directory holding the shared journal file.
    :return: An Optuna storage object.
    """
    makedirs(dir_path)
    journal_path = os.path.join(dir_path, HYPEROPT_JOURNAL_FILE_NAME)

    return JournalStorage(JournalFileBackend(journal_path, lock_obj=JournalFileOpenLock(journal_path)))


def create_study(
    storage: JournalStorage,
    study_name: str,
    minimize_score: bool,
    space: Dict[str, SearchParameter],
    sampler: optuna.samplers.BaseSampler,
) -> optuna.Study:
    """
    Loads the shared study, creating it if this is the first instance to reach it.

    The search space is recorded on the study the first time it is created and checked on
    every later load, so that instances sharing a journal file cannot silently combine
    trials drawn from different search spaces.

    :param storage: The shared Optuna storage, as returned by :func:`build_storage`.
    :param study_name: The name of the study within the storage.
    :param minimize_score: Whether a lower score is a better score.
    :param space: The search space, as returned by :func:`build_search_space`.
    :param sampler: The sampler used to choose parameters for the next trial.
    :return: An Optuna study object.
    """
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="minimize" if minimize_score else "maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    search_parameters = sorted(space.keys())
    loaded_parameters = study.user_attrs.get("search_parameters")
    if loaded_parameters is None:
        study.set_user_attr("search_parameters", search_parameters)
    elif sorted(loaded_parameters) != search_parameters:
        raise ValueError(
            f"A loaded Optuna study must be searching over the same parameters as the \
                hyperparameter optimization job. The loaded study covered variation in the parameters \
                {set(loaded_parameters)}. The current search is over the parameters {set(space.keys())}."
        )

    return study


def load_manual_trials(
    manual_trials_dirs: List[str],
    space: Dict[str, SearchParameter],
    hyperopt_args: HyperoptArgs,
) -> List[optuna.trial.FrozenTrial]:
    """
    Function for loading in manual training runs as trials for inclusion in hyperparameter search.
    Trials must be consistent with trials that would be generated in hyperparameter optimization.
    Parameters that are part of the search space do not have to match, but all others do.

    :param manual_trials_dirs: A list of paths to save directories for the manual trials, as would include test_scores.csv and args.json.
    :param space: The search space, as returned by :func:`build_search_space`.
    :param hyperopt_args: The arguments for the hyperparameter optimization job.
    :return: A list of completed Optuna trials, ready to be added to a study.
    """
    param_keys = list(space.keys())

    # Non-extensive list of arguments that need to match between the manual trials and the search space.
    matching_args = [
        ("number_of_molecules", None),
        ("aggregation", "aggregation"),
        ("num_folds", None),
        ("ensemble_size", None),
        ("max_lr", "max_lr"),
        ("init_lr", "init_lr_ratio"),
        ("final_lr", "final_lr_ratio"),
        ("activation", "activation"),
        ("metric", None),
        ("bias", None),
        ("epochs", None),
        ("explicit_h", None),
        ("adding_h", None),
        ("reaction", None),
        ("split_type", None),
        ("warmup_epochs", "warmup_epochs"),
        ("aggregation_norm", "aggregation_norm"),
        ("batch_size", "batch_size"),
        ("depth", "depth"),
        ("dropout", "dropout"),
        ("ffn_num_layers", "ffn_num_layers"),
        ("dataset_type", None),
        ("multiclass_num_classes", None),
        ("features_generator", None),
        ("no_features_scaling", None),
        ("features_only", None),
        ("split_sizes", None),
        ("weight_decay", "weight_decay"),
    ]

    manual_trials = []
    for trial_dir in manual_trials_dirs:

        # Extract trial data from test_scores.csv
        with open(os.path.join(trial_dir, "test_scores.csv")) as f:
            reader = csv.reader(f)
            next(reader)
            read_line = next(reader)
        mean_score = float(read_line[1])
        std_score = float(read_line[2])

        # Extract argument data from args.json
        with open(os.path.join(trial_dir, "args.json")) as f:
            trial_args = json.load(f)

        # Check for differences in manual trials and hyperopt space
        if "linked_hidden_size" in param_keys:
            if trial_args["hidden_size"] != trial_args["ffn_hidden_size"]:
                raise ValueError(
                    f'The manual trial in {trial_dir} has a hidden_size {trial_args["hidden_size"]} '
                    f'that does not match its ffn_hidden_size {trial_args["ffn_hidden_size"]}, as it would in hyperparameter search.'
                )
        elif "hidden_size" not in param_keys or "ffn_hidden_size" not in param_keys:
            if "hidden_size" not in param_keys:
                if getattr(hyperopt_args, "hidden_size") != trial_args["hidden_size"]:
                    raise ValueError(
                        f"Manual trial {trial_dir} has different training argument hidden_size than the hyperparameter optimization search trials."
                    )
            if "ffn_hidden_size" not in param_keys:
                if (
                    getattr(hyperopt_args, "ffn_hidden_size")
                    != trial_args["ffn_hidden_size"]
                ):
                    raise ValueError(
                        f"Manual trial {trial_dir} has different training argument ffn_hidden_size than the hyperparameter optimization search trials."
                    )

        for arg, space_parameter in matching_args:
            if space_parameter not in param_keys:
                if getattr(hyperopt_args, arg) != trial_args[arg]:
                    raise ValueError(
                        f"Manual trial {trial_dir} has different training argument {arg} than the hyperparameter optimization search trials."
                    )

        # Construct the hyperparameters of the trial, and the Optuna parameters that would produce them
        hyperparams = {}
        params = {}
        distributions = {}
        for key in param_keys:
            if key == "init_lr_ratio":
                value = trial_args["init_lr"] / trial_args["max_lr"]
            elif key == "final_lr_ratio":
                value = trial_args["final_lr"] / trial_args["max_lr"]
            elif key == "linked_hidden_size":
                value = trial_args["hidden_size"]
            else:
                value = trial_args[key]
            hyperparams[key] = value
            params.update(space[key].encode(key, value))
            distributions.update(space[key].distributions(key))

        # Conditional parameters only record the branch that was taken
        distributions = {key: value for key, value in distributions.items() if key in params}

        manual_trials.append(
            optuna.trial.create_trial(
                value=mean_score,
                params=params,
                distributions=distributions,
                user_attrs={
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "hyperparams": hyperparams,
                    "num_params": 0,
                    "manual_trial_dir": trial_dir,
                },
            )
        )

    return manual_trials


def add_manual_trials(
    study: optuna.Study,
    manual_trials: List[optuna.trial.FrozenTrial],
    logger: logging.Logger = None,
) -> None:
    """
    Adds manual trials to a shared study, at most once for the lifetime of that study.

    Because the study is shared between instances through the journal file, the manual trials
    only need to be contributed by whichever instance reaches the study first.

    :param study: The Optuna study to add the trials to.
    :param manual_trials: The manual trials, as returned by :func:`load_manual_trials`.
    :param logger: A logger for recording what was added.
    """
    info = print if logger is None else logger.info

    if study.user_attrs.get("manual_trials_added", False):
        info(
            f"{len(manual_trials)} manual trials were already added to the study by another "
            "hyperparameter optimization instance and are not being added again."
        )
        return

    study.set_user_attr("manual_trials_added", True)
    for trial in manual_trials:
        study.add_trial(trial)
    info(f"{len(manual_trials)} manual trials included in hyperparameter search.")


def get_completed_trials(study: optuna.Study) -> List[optuna.trial.FrozenTrial]:
    """
    Returns the trials of a study that completed with a usable score.

    :param study: The Optuna study to read trials from.
    :return: A list of completed trials with a non-nan mean score.
    """
    return [
        trial
        for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if not np.isnan(trial.user_attrs.get("mean_score", float("nan")))
    ]


def save_config(config_path: str, hyperparams_dict: dict, max_lr: float) -> None:
    """
    Saves the hyperparameters for the best trial to a config json file.

    :param config_path: File path for the config json file.
    :param hyperparams_dict: A dictionary of hyperparameters found during the search.
    :param max_lr: The maximum learning rate value, to be used if not a search parameter.
    """
    makedirs(config_path, isfile=True)

    save_dict = {}

    for key in hyperparams_dict:
        if key == "linked_hidden_size":
            save_dict["hidden_size"] = hyperparams_dict["linked_hidden_size"]
            save_dict["ffn_hidden_size"] = hyperparams_dict["linked_hidden_size"]
        elif key == "init_lr_ratio":
            if "max_lr" not in hyperparams_dict:
                save_dict["init_lr"] = hyperparams_dict[key] * max_lr
            else:
                save_dict["init_lr"] = (
                    hyperparams_dict[key] * hyperparams_dict["max_lr"]
                )
        elif key == "final_lr_ratio":
            if "max_lr" not in hyperparams_dict:
                save_dict["final_lr"] = hyperparams_dict[key] * max_lr
            else:
                save_dict["final_lr"] = (
                    hyperparams_dict[key] * hyperparams_dict["max_lr"]
                )
        else:
            save_dict[key] = hyperparams_dict[key]

    with open(config_path, "w") as f:
        json.dump(save_dict, f, indent=4, sort_keys=True)
