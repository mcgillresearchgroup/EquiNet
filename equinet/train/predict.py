from typing import List

import numpy as np
import torch
from tqdm import tqdm
from contextlib import nullcontext

from equinet.data import MoleculeDataLoader, MoleculeDataset, StandardScaler, AtomBondScaler
from equinet.models import MoleculeModel
from equinet.nn_utils import activate_dropout


def predict(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    disable_progress_bar: bool = False,
    scaler: StandardScaler = None,
    atom_bond_scaler: AtomBondScaler = None,
    return_unc_parameters: bool = False,
    dropout_prob: float = 0.0,
) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~equinet.models.model.MoleculeModel`.
    :param data_loader: A :class:`~equinet.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~equinet.features.scaler.StandardScaler` object fit on the training targets.
    :param atom_bond_scaler: A :class:`~equinet.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param return_unc_parameters: A bool indicating whether additional uncertainty parameters would be returned alongside the mean predictions.
    :param dropout_prob: For use during uncertainty prediction only. The propout probability used in generating a dropout ensemble.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks. If returning uncertainty parameters as well,
        it is a tuple of lists of lists, of a length depending on how many uncertainty parameters are appropriate for the loss function.
    """
    model.eval()

    # Activate dropout layers to work during inference for uncertainty estimation
    if dropout_prob > 0.0:

        def activate_dropout_(model):
            return activate_dropout(model, dropout_prob)

        model.apply(activate_dropout_)

    preds = []

    var, lambdas, alphas, betas = [], [], [], []  # only used if returning uncertainty parameters

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch = batch.batch_graph()
        features_batch = batch.features()
        atom_descriptors_batch = batch.atom_descriptors()
        atom_features_batch = batch.atom_features()
        bond_descriptors_batch = batch.bond_descriptors()
        bond_features_batch = batch.bond_features()
        constraints_batch = batch.constraints()
        hybrid_model_features_batch = batch.hybrid_model_features()

        bond_types_batch = None

        # set whether gradients are needed during prediction
        if model.vle == "freestyle":  
            predict_gradient = True
        else:
            predict_gradient = False

        with (nullcontext() if predict_gradient else torch.no_grad()):
            batch_preds = model(
                mol_batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
                constraints_batch,
                bond_types_batch,
                hybrid_model_features_batch,
            )

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if model.vle is not None and model.vle != "basic":
            # y1 y2 logP g1 g2 log10p1sat log10p2sat
            # unscale logP, log10p1sat, log10p2sat
            batch_preds[:,[2,5,6]] = batch_preds[:,[2,5,6]] * scaler.stds[2] + scaler.means[2]
        elif scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
            if model.loss_function == "mve":
                batch_var = batch_var * scaler.stds**2
            elif model.loss_function == "evidential":
                batch_betas = batch_betas * scaler.stds**2

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
