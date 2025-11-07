import logging
from typing import Callable

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from torch import autograd

from equinet.args import TrainArgs
from equinet.data import MoleculeDataLoader, MoleculeDataset, AtomBondScaler
from equinet.models import MoleculeModel
from equinet.nn_utils import compute_gnorm, compute_pnorm, NoamLR
from equinet.utils import print_nan_diagnostic


def train(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    loss_func: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    args: TrainArgs,
    n_iter: int = 0,
    atom_bond_scaler: AtomBondScaler = None,
    logger: logging.Logger = None,
    writer: SummaryWriter = None,
) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~equinet.models.model.MoleculeModel`.
    :param data_loader: A :class:`~equinet.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~equinet.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param atom_bond_scaler: A :class:`~equinet.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()

    if model.is_atom_bond_targets:
        loss_sum, iter_count = [0]*(len(args.atom_targets) + len(args.bond_targets)), 0
    else:
        loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch, mask_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch, constraints_batch, data_weights_batch, hybrid_model_features_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.mask(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_descriptors(), batch.bond_features(), batch.constraints(), batch.data_weights(), batch.hybrid_model_features()

        mask_batch = np.transpose(mask_batch).tolist()
        masks = torch.tensor(mask_batch, dtype=torch.bool)  # shape(batch, tasks)
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch])  # shape(batch, tasks)

        if args.target_weights is not None:
            target_weights = torch.tensor(args.target_weights).unsqueeze(0)  # shape(1,tasks)
        else:
            target_weights = torch.ones(targets.shape[1]).unsqueeze(0)
        data_weights = torch.tensor(data_weights_batch).unsqueeze(1)  # shape(batch,1)

        constraints_batch = None
        bond_types_batch = None

        # Run model
        model.zero_grad()
        preds = model(
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
        if args.self_activity_lambda > 0:
            preds, regularization = preds
        else:
            regularization = 0

        # Move tensors to correct device
        torch_device = args.device
        masks = masks.to(torch_device)
        targets = targets.to(torch_device)
        target_weights = target_weights.to(torch_device)
        data_weights = data_weights.to(torch_device)

        if args.fugacity_balance:
            hybrid_model_features_batch = torch.tensor(np.array(hybrid_model_features_batch)).to(torch_device)
            x1_not_zero = hybrid_model_features_batch[:,[0]] != 0
            x2_not_zero = hybrid_model_features_batch[:,[1]] != 0
            g1_present = masks[:,[3]]
            g2_present = masks[:,[4]]
            masks = torch.cat([
                x1_not_zero & ~(g1_present | g2_present), # x1 not zero and g1 and g2 not present
                x2_not_zero & ~(g1_present | g2_present), # x2 not zero and g1 and g2 not present
                g1_present,
                g2_present,
            ], dim=1).bool()

        # Calculate losses
        if args.loss_function == "squared_log_fugacity_difference":
            loss = loss_func(preds, targets, hybrid_model_features_batch, masks) * data_weights
        else:
            loss = loss_func(preds, targets) * target_weights * data_weights
            loss = torch.where(masks, loss, 0)

        if (loss.isnan().any() or loss.isinf().any() or preds.isnan().any() or preds.isinf().any()) and not args.hyperparateter_optimization:
            print_nan_diagnostic(batch, model, args, loss, preds, logger)
            raise ValueError("Loss or preds contain NaNs or infs")

        loss = loss.sum() / masks.sum()

        loss = loss + regularization

        loss_sum += loss.item()
        iter_count += 1

        loss.backward()

        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            if model.is_atom_bond_targets:
                loss_avg = sum(loss_sum) / iter_count
                loss_sum, iter_count = [0]*(len(args.atom_targets) + len(args.bond_targets)), 0
            else:
                loss_avg = loss_sum / iter_count
                loss_sum = iter_count = 0

            lrs_str = ", ".join(f"lr_{i} = {lr:.4e}" for i, lr in enumerate(lrs))
            debug(f"Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}")

            if writer is not None:
                writer.add_scalar("train_loss", loss_avg, n_iter)
                writer.add_scalar("param_norm", pnorm, n_iter)
                writer.add_scalar("gradient_norm", gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f"learning_rate_{i}", lr, n_iter)

    return n_iter
