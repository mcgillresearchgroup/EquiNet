import torch
import torch.nn as nn
import numpy as np


def forward_vp(
    vp: str,
    output: torch.Tensor,
    temperature: torch.Tensor,
    Tc: torch.Tensor = None,
    log10Pc: torch.Tensor = None,
):
    """
    
    """
    if vp == "basic":
        pass
    elif vp == "antoine":
        antoine_a, antoine_b, antoine_c = torch.chunk(output, 3, dim=1)
        antoine_b = nn.functional.softplus(antoine_b)
        output = antoine_a - (antoine_b / nn.functional.softplus(antoine_c + temperature, beta=0.1))
    return output


def get_vp_parameter_names(
        vp: str,
        molecule_id: int = None,
):
    """
    Get the coefficients for the vapor pressure model. If there are multiple molecules
    in a mixture for vle prediction, the coefficients are suffixed with the molecule ID.
    """
    if vp == "antoine":
        names = ["antoine_a", "antoine_b", "antoine_c"]
    else:
        raise NotImplementedError(f"Vapor pressure model {vp} not supported")
    if molecule_id is not None:
        names = [f"{name}_{molecule_id}" for name in names]
    return names


def unscale_vp_parameters(
        parameters: np.ndarray,
        target_scaler,
        hybrid_model_features_scaler,
        vle: str,
        vp: str,
):
    """
    Unscale the vapor pressure parameters.
    """
    if vle is not None:
        mean_p, std_p = target_scaler.means[2], target_scaler.stds[2]
        scale_t = hybrid_model_features_scaler.stds[2]
    else:
        mean_p, std_p = target_scaler.means[0], target_scaler.stds[0]
        scale_t = hybrid_model_features_scaler.stds[0]
    if vp == "antoine":
        antoine_a, antoine_b, antoine_c = np.split(parameters, 3, axis=1)
        antoine_a = std_p * antoine_a - mean_p
        antoine_b = std_p * scale_t * antoine_b
        antoine_c = scale_t * antoine_c
        parameters = np.concatenate([antoine_a, antoine_b, antoine_c], axis=1)
    else:
        raise NotImplementedError(f"Vapor pressure model {vp} not supported for unscaling vp parameters")
    return parameters