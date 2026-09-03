"""Optimizes hyperparameters using Bayesian optimization."""

from typing import Dict, Union
import os

import numpy as np
import optuna

from equinet.args import HyperoptArgs
from equinet.constants import HYPEROPT_LOGGER_NAME
from equinet.models import MoleculeModel
from equinet.nn_utils import param_count
from equinet.train import cross_validate, run_training
from equinet.utils import create_logger, makedirs, timeit
from equinet.optuna_utils import add_manual_trials, build_search_space, build_storage, \
    build_trial_args, create_study, get_completed_trials, load_manual_trials, \
    save_config, suggest_hyperparameters


@timeit(logger_name=HYPEROPT_LOGGER_NAME)
def hyperopt(args: HyperoptArgs) -> None:
    """
    Runs hyperparameter optimization on a EquiNet model.

    Hyperparameter optimization optimizes the following parameters:

    * :code:`hidden_size`: The hidden size of the neural network layers is selected from {300, 400, ..., 2400}
    * :code:`depth`: The number of message passing iterations is selected from {2, 3, 4, 5, 6}
    * :code:`dropout`: The dropout probability is selected from {0.0, 0.05, ..., 0.4}
    * :code:`ffn_num_layers`: The number of feed-forward layers after message passing is selected from {1, 2, 3}

    Trials are stored in an Optuna study backed by a journal file in
    :code:`args.hyperopt_checkpoint_dir`. Any number of instances pointed at the same directory
    contribute to and read from that one study, so the search can be parallelized simply by
    launching more jobs with a shared checkpoint directory.

    The best set of hyperparameters is saved as a JSON file to :code:`args.config_save_path`.

    :param args: A :class:`~equinet.args.HyperoptArgs` object containing arguments for hyperparameter
                 optimization in addition to all arguments needed for training.
    """
    # Create logger
    logger = create_logger(name=HYPEROPT_LOGGER_NAME, save_dir=args.log_dir, quiet=True)

    # Every trial is already reported through the logger above, and reloading the shared study
    # once per iteration makes Optuna's own progress logging repetitive.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Build search space
    logger.info(f"Creating search space using parameters {args.search_parameters}.")
    space = build_search_space(
        search_parameters=args.search_parameters, train_epochs=args.epochs
    )

    makedirs(args.hyperopt_checkpoint_dir)

    # Set up the shared file-based storage that lets parallel instances see each other's trials
    storage = build_storage(args.hyperopt_checkpoint_dir)

    # Load in manual trials
    if args.manual_trial_dirs is not None:
        manual_trials = load_manual_trials(
            manual_trials_dirs=args.manual_trial_dirs,
            space=space,
            hyperopt_args=args,
        )
    else:
        manual_trials = None
        logger.info("No manual trials loaded as part of hyperparameter search")

    # Define hyperparameter optimization
    def objective(trial: optuna.Trial) -> float:
        logger.info(f"Initiating trial {trial.number}")
        # suggest_hyperparameters returns values keyed by the argument names they set, so they can
        # be applied directly. The args are then rebuilt from the originally parsed namespace with
        # them in place, so that process_args derives everything from this trial's values rather
        # than from the ones the job was launched with.
        hyperparams: Dict[str, Union[int, float]] = suggest_hyperparameters(trial, space, args)

        overrides: Dict[str, Union[int, float]] = dict(hyperparams)

        if args.save_dir is not None:
            folder_name = f"trial_{trial.number}"
            overrides["save_dir"] = os.path.join(args.save_dir, folder_name)

        hyper_args = build_trial_args(args, overrides)

        # Cross validate
        mean_score, std_score = cross_validate(args=hyper_args, train_func=run_training)

        # Record results
        temp_model = MoleculeModel(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f"Trial {trial.number} results")
        logger.info(hyperparams)
        logger.info(f"num params: {num_params:,}")
        logger.info(f"{mean_score} +/- {std_score} {hyper_args.metric}")

        # Deal with nan
        if np.isnan(mean_score):
            if hyper_args.dataset_type == "classification":
                mean_score = 0
            else:
                raise ValueError(
                    "Can't handle nan score for non-classification dataset."
                )

        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("std_score", std_score)
        trial.set_user_attr("hyperparams", hyperparams)
        trial.set_user_attr("num_params", num_params)

        return mean_score

    # One sampler for the lifetime of this instance. Unseeded by default, so that instances sharing
    # a checkpoint directory draw independent random streams instead of all proposing the same
    # starting parameters. constant_liar additionally keeps them from proposing the same point while
    # another instance still has that trial running.
    if args.hyperopt_seed is not None:
        logger.info(
            f"Sampling parameters with seed {args.hyperopt_seed}. This reproduces a single-instance "
            "search; parallel instances sharing a checkpoint directory should each be left unseeded."
        )
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.startup_random_iters,
        constant_liar=True,
        seed=args.hyperopt_seed,
    )

    # Iterate over a number of trials
    for i in range(args.num_iters):
        # Reload the study at each iteration so that trials finished by parallel instances are included
        study = create_study(
            storage=storage,
            study_name=args.hyperopt_study_name,
            minimize_score=args.minimize_score,
            space=space,
            sampler=sampler,
        )

        if manual_trials is not None:
            add_manual_trials(study=study, manual_trials=manual_trials, logger=logger)
            manual_trials = None

        num_trials = len(study.trials)
        if num_trials >= args.num_iters:
            break

        # Log the start of the trial
        logger.info(f"Loaded {num_trials} previous trials")
        if num_trials < args.startup_random_iters:
            random_remaining = args.startup_random_iters - num_trials
            logger.info(
                f"Parameters assigned with random search, {random_remaining} random trials remaining"
            )
        else:
            logger.info(f"Parameters assigned with TPE directed search")

        study.optimize(objective, n_trials=1)

    # Report best result
    study = create_study(
        storage=storage,
        study_name=args.hyperopt_study_name,
        minimize_score=args.minimize_score,
        space=space,
        sampler=sampler,
    )
    results = get_completed_trials(study)
    if len(results) == 0:
        raise ValueError("No trials completed with a usable score, so no best trial can be reported.")
    best_trial = min(
        results,
        key=lambda trial: (1 if args.minimize_score else -1) * trial.user_attrs["mean_score"],
    )
    best_result = best_trial.user_attrs
    if "manual_trial_dir" in best_result:
        logger.info(f'Best trial, number {best_trial.number}, '
                    f'a manual trial loaded from {best_result["manual_trial_dir"]}')
    else:
        logger.info(f"Best trial, number {best_trial.number}")
    logger.info(best_result["hyperparams"])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(
        f'{best_result["mean_score"]} +/- {best_result["std_score"]} {args.metric}'
    )

    # Save best hyperparameter settings as JSON config file
    save_config(
        config_path=args.config_save_path,
        hyperparams_dict=best_result["hyperparams"],
    )


def equinet_hyperopt() -> None:
    """Runs hyperparameter optimization for a EquiNet model.

    This is the entry point for the command line command :code:`equinet_hyperopt`.
    """
    hyperopt(args=HyperoptArgs().parse_args())
