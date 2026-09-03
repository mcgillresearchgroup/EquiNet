"""Utilities supporting Optuna-based hyperparameter optimization."""

import csv
import json
import logging
import os
from copy import deepcopy
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


class ZeroOrValueParameter(SearchParameter):
    """
    A search parameter that is either switched off, and exactly zero, or switched on and drawn
    from a positive range.

    Whether the parameter is used at all is itself part of the search. This is a conditional
    search space: the magnitude is only sampled on the nonzero branch, so trials that switch the
    parameter off never record a magnitude parameter at all, rather than recording a zero that
    the sampler would treat as an ordinary point of the range.
    """

    def __init__(self, low: float, high: float, log: bool = False):
        self.low = low
        self.high = high
        self.log = log

    def suggest(self, name: str, trial: optuna.Trial) -> float:
        if trial.suggest_categorical(f"{name}_nonzero", [False, True]):
            return trial.suggest_float(f"{name}_magnitude", self.low, self.high, log=self.log)
        return 0.0

    def distributions(self, name: str) -> Dict[str, BaseDistribution]:
        return {
            f"{name}_nonzero": CategoricalDistribution([False, True]),
            f"{name}_magnitude": FloatDistribution(self.low, self.high, log=self.log),
        }

    def encode(self, name: str, value: Any) -> Dict[str, Any]:
        if value == 0:
            return {f"{name}_nonzero": False}
        return {f"{name}_nonzero": True, f"{name}_magnitude": value}


class BoundedLogFloatParameter(SearchParameter):
    """
    A log-uniform search parameter whose upper bound is decided by another parameter.

    ``init_lr`` and ``final_lr`` are searched directly over their own range rather than as a
    fraction of ``max_lr``, but neither is meaningful above the maximum learning rate, so the
    range is truncated at whatever ``max_lr`` the trial is using. Optuna records the bounds it
    actually sampled from with each trial, so the bound is allowed to differ from trial to trial.
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def _high(self, upper_bound: float) -> float:
        # A max_lr at or below the floor collapses the range to a single point, which is allowed
        return max(self.low, min(self.high, upper_bound))

    def suggest(self, name: str, trial: optuna.Trial, upper_bound: float = None) -> float:
        return trial.suggest_float(name, self.low, self._high(upper_bound), log=True)

    def distributions(self, name: str, upper_bound: float = None) -> Dict[str, BaseDistribution]:
        return {name: FloatDistribution(self.low, self._high(upper_bound), log=True)}


# Parameters that are only searched, or whose range is only known, once another parameter has
# been sampled. They are suggested after the rest, in this order.
DEPENDENT_PARAMETERS = ("aggregation_norm", "init_lr", "final_lr")


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
        # vle, vp and fugacity_balance are deliberately absent. Trials are built by setting
        # attributes on an already processed args object, so parse-time derivations do not re-run,
        # and those three drive derivations: vle sets number_of_molecules and mpn_shared, and
        # fugacity_balance selects the squared_log_fugacity_difference loss, which cannot be set
        # any other way. Searching over them silently trains against the wrong loss.
        "wohl_order": IntParameter(low=3, high=5),
        "self_activity_correction": CategoricalParameter([True, False]),
        "self_activity_lambda": ZeroOrValueParameter(low=1e-5, high=1e-1, log=True),
        "aggregation": CategoricalParameter(["mean", "sum", "norm"]),
        # Only reached when the aggregation in use is "norm"; see suggest_hyperparameters
        "aggregation_norm": IntParameter(low=1, high=200),
        "batch_size": IntParameter(low=5, high=200, step=5),
        "depth": IntParameter(low=2, high=6),
        "dropout": ZeroOrValueParameter(low=1e-2, high=4e-1, log=True),
        "ffn_hidden_size": IntParameter(low=300, high=2400, step=100),
        "ffn_num_layers": IntParameter(low=2, high=6),
        "hidden_size": IntParameter(low=300, high=2400, step=100),
        # hidden_size and ffn_hidden_size constrained to one shared value, rather than a
        # parameter of its own; see suggest_hyperparameters
        "linked_hidden_size": IntParameter(low=300, high=2400, step=100),
        "max_lr": FloatParameter(low=1e-6, high=1e-2, log=True),
        # Searched over their own range, truncated at the max_lr the trial is using
        "init_lr": BoundedLogFloatParameter(low=1e-6, high=1e-2),
        "final_lr": BoundedLogFloatParameter(low=1e-6, high=1e-2),
        "weight_decay": ZeroOrValueParameter(low=1e-6, high=1e-1, log=True),
    }  # TODO add any new parameters here
    if train_epochs is not None:
        available_spaces["warmup_epochs"] = IntParameter(
            low=1, high=max(1, train_epochs // 2)
        )

    space = {}
    for key in search_parameters:
        space[key] = available_spaces[key]

    return space


def suggest_hyperparameters(
    trial: optuna.Trial, space: Dict[str, SearchParameter], args: HyperoptArgs
) -> Dict[str, Any]:
    """
    Samples one set of hyperparameters from the search space.

    Most parameters are sampled independently, in a fixed order so that a seeded sampler draws
    them the same way on every run. The rest depend on a parameter sampled before them: the
    aggregation norm is only meaningful for norm aggregation, and the initial and final learning
    rates are bounded above by the maximum learning rate. Where such a parameter is not itself
    being searched, the value the job was launched with is used instead.

    :param trial: The Optuna trial to sample from.
    :param space: The search space, as returned by :func:`build_search_space`.
    :param args: The arguments of the hyperparameter optimization job, for the values of any
                 parameters that are depended on but not searched.
    :return: A dictionary keyed by the argument names of the sampled values.
    """
    hyperparams: Dict[str, Any] = {}

    for key in sorted(space):
        if key in DEPENDENT_PARAMETERS or key == "linked_hidden_size":
            continue
        hyperparams[key] = space[key].suggest(key, trial)

    # One searched value shared by both hidden sizes, recorded under the name it is searched by
    if "linked_hidden_size" in space:
        linked_hidden_size = space["linked_hidden_size"].suggest("linked_hidden_size", trial)
        hyperparams["hidden_size"] = linked_hidden_size
        hyperparams["ffn_hidden_size"] = linked_hidden_size

    if "aggregation_norm" in space:
        aggregation = hyperparams.get("aggregation", args.aggregation)
        if aggregation == "norm":
            hyperparams["aggregation_norm"] = space["aggregation_norm"].suggest(
                "aggregation_norm", trial
            )

    max_lr = hyperparams.get("max_lr", args.max_lr)
    for key in ("init_lr", "final_lr"):
        if key in space:
            hyperparams[key] = space[key].suggest(key, trial, upper_bound=max_lr)

    return hyperparams


def build_trial_args(args: HyperoptArgs, overrides: Dict[str, Any]) -> HyperoptArgs:
    """
    Builds the arguments for a single trial.

    The trial's values are applied to the arguments as they were originally parsed, and
    ``process_args`` is then run over the result, so that everything derived from a searched
    argument is derived from the value that trial actually uses. Assigning the values onto an
    already processed arguments object instead would leave those derivations stale: they run once,
    when the job is launched, and would still reflect the arguments it was launched with.

    This is a function rather than a method on :class:`~equinet.args.HyperoptArgs` because
    ``Tap.as_dict`` reports any attribute name it does not recognise from the base class. A method
    would be reported as a bound method, saved into every model checkpoint, and would carry the
    whole arguments object into the checkpoint with it.

    :param args: The processed arguments of the hyperparameter optimization job.
    :param overrides: The argument values for this trial, keyed by argument name.
    :return: A new arguments object, processed and ready to be trained with.
    """
    trial_args = deepcopy(args)
    trial_args.__dict__.update(deepcopy(args._raw_arg_values))

    for key, value in overrides.items():
        setattr(trial_args, key, value)

    trial_args.process_args()

    return trial_args


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
        ("init_lr", "init_lr"),
        ("final_lr", "final_lr"),
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

        # Construct the hyperparameters of the trial, and the Optuna parameters that would produce
        # them. This mirrors suggest_hyperparameters: the dependent parameters are only recorded
        # when the trial they came from actually used them, and the learning rates carry the
        # bound that the trial's own max_lr implies.
        hyperparams = {}
        params = {}
        distributions = {}

        def record(key: str, name: str, value: Any, **kwargs) -> None:
            """Records one value under its argument name, and under the name it is searched by."""
            hyperparams[key] = value
            params.update(space[name].encode(name, value))
            distributions.update(space[name].distributions(name, **kwargs))

        for key in param_keys:
            if key in DEPENDENT_PARAMETERS or key == "linked_hidden_size":
                continue
            record(key, key, trial_args[key])

        if "linked_hidden_size" in param_keys:
            record("hidden_size", "linked_hidden_size", trial_args["hidden_size"])
            hyperparams["ffn_hidden_size"] = trial_args["hidden_size"]

        if "aggregation_norm" in param_keys:
            if trial_args["aggregation"] == "norm":
                record("aggregation_norm", "aggregation_norm", trial_args["aggregation_norm"])

        max_lr = trial_args["max_lr"]
        for key in ("init_lr", "final_lr"):
            if key in param_keys:
                if trial_args[key] > max_lr:
                    raise ValueError(
                        f"The manual trial in {trial_dir} has a {key} of {trial_args[key]} that is "
                        f"greater than its max_lr of {max_lr}, which the hyperparameter search "
                        f"would never produce."
                    )
                record(key, key, trial_args[key], upper_bound=max_lr)

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


def save_config(config_path: str, hyperparams_dict: dict) -> None:
    """
    Saves the hyperparameters for the best trial to a config json file.

    The hyperparameters are already keyed by the training argument names they set, so they are
    written out as they are.

    :param config_path: File path for the config json file.
    :param hyperparams_dict: A dictionary of hyperparameters found during the search.
    """
    makedirs(config_path, isfile=True)

    with open(config_path, "w") as f:
        json.dump(dict(hyperparams_dict), f, indent=4, sort_keys=True)
