from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from equinet.nn_utils import get_activation_function

def build_ffn(
    first_linear_dim: int,
    hidden_size: int,
    num_layers: int,
    output_size: int,
    dropout: float,
    activation: str,
    dataset_type: str = None,
    spectra_activation: str = None,
) -> nn.Sequential:
    """
    Returns an `nn.Sequential` object of FFN layers.

    :param first_linear_dim: Dimensionality of fisrt layer.
    :param hidden_size: Dimensionality of hidden layers.
    :param num_layers: Number of layers in FFN.
    :param output_size: The size of output.
    :param dropout: Dropout probability.
    :param activation: Activation function.
    :param dataset_type: Type of dataset.
    :param spectra_activation: Activation function used in dataset_type spectra training to constrain outputs to be positive.
    """
    activation = get_activation_function(activation)

    if num_layers == 1:
        layers = [
            nn.Dropout(dropout),
            nn.Linear(first_linear_dim, output_size)
        ]
    else:
        layers = [
            nn.Dropout(dropout),
            nn.Linear(first_linear_dim, hidden_size)
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            ])
        layers.extend([
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        ])

    return nn.Sequential(*layers)


def binary_equivariant_readout(
    encoding_1: torch.Tensor,
    encoding_2: torch.Tensor,
    features: torch.Tensor,
    readout: nn.Module,
    output_equivariant_pairs: List[Tuple[int, int]] = [],
    features_equivariant_pairs: List[Tuple[int, int]] = [],
) -> torch.Tensor:
    """
    """
    features_1 = features
    features_2 = features.clone()
    for i, j in features_equivariant_pairs:
        features_2[:, [i, j]] = features_1[:, [j, i]]
    output_1 = readout(torch.cat([encoding_1, encoding_2, features_1], dim=1))
    output_2 = readout(torch.cat([encoding_2, encoding_1, features_2], dim=1))
    for i, j in output_equivariant_pairs:
        output_2[:, [i, j]] = output_2[:, [j, i]]

    average = (output_1 + output_2) / 2
    return average


def binary_permutation_readout(
    encoding_1: torch.Tensor,
    encoding_2: torch.Tensor,
    features: torch.Tensor,
    readout: nn.Module,
    features_equivariant_pairs: List[Tuple[int, int]] = [],
) -> tuple[torch.Tensor]:
    """
    """
    # make an output for each permutation of the two encodings and return them all [out11, out12, out21, out22]
    features_12 = features
    features_11 = features.clone()
    features_21 = features.clone()
    features_22 = features.clone()
    for i, j in features_equivariant_pairs:
        features_21[:, [i, j]] = features_12[:, [j, i]] # swap the features
        features_11[:, j] = features_12[:, i] # all molecule 1 features become molecule 2 features
        features_22[:, i] = features_12[:, j] # all molecule 2 features become molecule 1 features

    output_12 = readout(torch.cat([encoding_1, encoding_2, features_12], dim=1))
    output_11 = readout(torch.cat([encoding_1, encoding_1, features_11], dim=1))
    output_21 = readout(torch.cat([encoding_2, encoding_1, features_21], dim=1))
    output_22 = readout(torch.cat([encoding_2, encoding_2, features_22], dim=1))

    return (output_11, output_12, output_21, output_22)

