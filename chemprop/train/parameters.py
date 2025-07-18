from typing import List, Union
import csv

import torch
import numpy as np
from tqdm import tqdm

from chemprop.args import ParameterArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, makedirs, timeit, load_scalers, update_prediction_args
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.features import set_reaction, set_explicit_h, set_adding_hs, set_keeping_atom_map, reset_featurization_parameters, set_extra_atom_fdim, set_extra_bond_fdim
from chemprop.models import MoleculeModel
from chemprop.models.vle import unscale_vle_parameters
from chemprop.models.vp import unscale_vp_parameters



def get_parameters(
        args: ParameterArgs,
) -> None:
    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[ParameterArgs, TrainArgs]

    reset_featurization_parameters()

    set_explicit_h(train_args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)

    test_data = get_data(path=args.test_path, smiles_columns=args.smiles_columns, target_columns=[], ignore_columns=[], skip_invalid_smiles=False,
                             args=args, store_row=True)

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        if index > 0:
            raise NotImplementedError('Only one model checkpoint is supported for parameter prediction')
        model = load_checkpoint(checkpoint_path, device=args.device)
        scaler, features_scaler, atom_descriptor_scaler, bond_descriptor_scaler, atom_bond_scaler, hybrid_model_features_scaler = load_scalers(args.checkpoint_paths[index])

        # Normalize features
        test_data.reset_features_and_targets()
        if features_scaler is not None:
            test_data.normalize_features(features_scaler)
        if hybrid_model_features_scaler is not None:
            test_data.normalize_hybrid_model_features(hybrid_model_features_scaler)

        # Predictions
        model.eval()
        parameters = []
        for batch in tqdm(test_data_loader, leave=False):
            batch: MoleculeDataset
            mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch, hybrid_model_features_batch = \
                batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_descriptors(), batch.bond_features(), batch.hybrid_model_features()

            with torch.no_grad():
                names, batch_parameters = model(
                    batch=mol_batch,
                    features_batch=features_batch,
                    hybrid_model_features_batch=hybrid_model_features_batch,
                    get_parameters=True,
                )
            batch_parameters = batch_parameters.cpu().tolist()
            parameters.extend(batch_parameters)

        parameters = np.array(parameters)
        if not args.internal_scaled_parameters:
            unscaled_parameters = []
            if args.vle not in [None, "basic", "activity"]:
                num_vle = model.vle_output_size
                if args.vle in ["wohl","nrtl-wohl"]:
                    num_vle += 2 # q coefficients
                if args.self_activity_correction:
                    num_vle *= 3
                vle_parameters = parameters[:, :num_vle]
                vle_parameters = unscale_vle_parameters(vle_parameters, hybrid_model_features_scaler, args.vle, args.wohl_order)
                unscaled_parameters.append(vle_parameters)
                if args.vp is not None and args.vp != "basic":
                    num_vp = model.vp_output_size
                    vp1_parameters = parameters[:, num_vle:num_vle+num_vp]
                    vp1_parameters = unscale_vp_parameters(vp1_parameters, scaler, hybrid_model_features_scaler, args.vle, args.vp)
                    unscaled_parameters.append(vp1_parameters)
                    vp2_parameters = parameters[:, num_vle+num_vp:]
                    vp2_parameters = unscale_vp_parameters(vp2_parameters, scaler, hybrid_model_features_scaler, args.vle, args.vp)
                    unscaled_parameters.append(vp2_parameters)
            elif args.vp not in [None, "basic"]:
                vp_parameters = unscale_vp_parameters(parameters, scaler, hybrid_model_features_scaler, args.vle, args.vp)
                unscaled_parameters.append(vp_parameters)
            parameters = np.concatenate(unscaled_parameters, axis=1)
    # Copy predictions over to full_data
    for index, datapoint in enumerate(test_data):
        row_preds = parameters[index]
        for i in range(len(names)):
            datapoint.row[names[i]] = row_preds[i]

    # Write predictions
    with open(args.preds_path, 'w', newline="") as f:
        writer = csv.DictWriter(f, fieldnames=args.smiles_columns+names, extrasaction='ignore')
        writer.writeheader()
        for datapoint in test_data:
            writer.writerow(datapoint.row)


def chemprop_parameters() -> None:
    """
    Parses Chemprop predicting arguments and returns the latent representation vectors for
    provided molecules, according to a previously trained model.
    """
    get_parameters(args=ParameterArgs().parse_args())
