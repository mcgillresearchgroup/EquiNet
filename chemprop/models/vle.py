import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def forward_vle_basic(
        output: torch.Tensor,
):
    """
    vle output calculation for the basic vle model
    """
    logit_y1, logit_y2, log10P, ln_gamma_1, ln_gamma_2, log10p1sat, log10p2sat = torch.chunk(output, 7, dim=1)
    ys = torch.softmax(torch.cat([logit_y1, logit_y2], axis=1), dim=1)
    output = torch.cat([ys, log10P, ln_gamma_1, ln_gamma_2, log10p1sat, log10p2sat], axis=1)
    return output


def forward_vle_activity(
        output: torch.Tensor,
):
    """
    vle output calculation for the activity coefficient model
    """
    ln_gamma_1 = output[:,[0]]
    ln_gamma_2 = output[:,[1]]
    return ln_gamma_1, ln_gamma_2


def get_activity_parameters(
        output: torch.Tensor,
        molecule_id: int = None,
):
    """
    Get the activity coefficients and their names
    """
    names = ['ln_gamma_1', 'ln_gamma_2']
    if molecule_id is not None:
        names = [f"{name}_{molecule_id}" for name in names]
    return names, output


def forward_vle_wohl(
        output: torch.Tensor,
        wohl_order: int,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        q_1: torch.Tensor,
        q_2: torch.Tensor,
):
    """
    """
    z_1 = q_1 * x_1 / (q_1 * x_1 + q_2 * x_2)
    z_2 = q_2 * x_2 / (q_1 * x_1 + q_2 * x_2)

    if wohl_order == 3:
        a12, a112, a122 = torch.chunk(output, 3, dim=1)
        ln_gamma_1 = \
            2 * a12 * z_2**2 * q_1 + \
            6 * a112 * z_1 * z_2**2 * q_1 - \
            3 * a122 * z_1 * z_2**2 * q_1 + \
            3 * a122 * z_2**3 * q_1
        ln_gamma_2 = \
            2 * a12 * z_1**2 * q_2 + \
            3 * a112 * z_1**3 * q_2 - \
            3 * a112 * z_1**2 * z_2 * q_2 + \
            6 * a122 * z_1**2 * z_2 * q_2
    elif wohl_order == 4:
        a12, a112, a122, a1112, a1122, a1222 = torch.chunk(output, 6, dim=1)
        ln_gamma_1 = \
            2 * a12 * z_2**2 * q_1 + \
            6 * a112 * z_1 * z_2**2 * q_1 - \
            3 * a122 * z_1 * z_2**2 * q_1 + \
            3 * a122 * z_2**3 * q_1 + \
            12 * a1112 * z_1**2 * q_1 * z_2**2 * q_2 + \
            4 * a1222 * q_1 * z_2**4 - \
            8 * a1222 * z_1 * q_1 * z_2**3 + \
            12 * a1122 * z_1 * q_1 * z_2**3 - \
            6 * a1122 * z_1**2 * q_1 * z_2**2
        ln_gamma_2 = \
            2 * a12 * z_1**2 * q_2 + \
            3 * a112 * z_1**3 * q_2 - \
            3 * a112 * z_1**2 * z_2 * q_2 + \
            6 * a122 * z_1**2 * z_2 * q_2 + \
            4 * a1112 * z_1**4 * q_2 - \
            8 * a1112 * z_1**3 * z_2 * q_2 + \
            12 * a1122 * z_1**2 * z_2**2 * q_2 + \
            12 * a1222 * z_1**3 * z_2 * q_2 - \
            6 * a1222 * z_1**2 * z_2**2 * q_2
    elif wohl_order == 5:
        a12, a112, a122, a1112, a1122, a1222, a11112, a11122, a11222, a12222 = torch.chunk(output, 10, dim=1)
        ln_gamma_1 = \
            2 * a12 * z_2**2 * q_1 + \
            6 * a112 * z_1 * z_2**2 * q_1 - \
            3 * a122 * z_1 * z_2**2 * q_1 + \
            3 * a122 * z_2**3 * q_1 + \
            12 * a1112 * z_1**2 * q_1 * z_2**2 * q_2 + \
            4 * a1222 * q_1 * z_2**4 - \
            8 * a1222 * z_1 * q_1 * z_2**3 + \
            12 * a1122 * z_1 * q_1 * z_2**3 - \
            6 * a1122 * z_1**2 * q_1 * z_2**2 + \
            20 * a11112 * z_1**3 * q_1 * z_2**2 + \
            30 * a11122 * z_1**2 * q_1 * z_2**3 - \
            10 * a11122 * z_1**3 * q_1 * z_2**2 + \
            20 * a11222 * z_1 * q_1 * z_2**4 - \
            20 * a11222 * z_1**2 * q_1 * z_2**3 + \
            5 * a12222 * q_1 * z_2**5 - \
            15 * a12222 * z_1 * q_1 * z_2**4
        ln_gamma_2 = \
            2 * a12 * z_1**2 * q_2 + \
            3 * a112 * z_1**3 * q_2 - \
            3 * a112 * z_1**2 * z_2 * q_2 + \
            6 * a122 * z_1**2 * z_2 * q_2 + \
            4 * a1112 * z_1**4 * q_2 - \
            8 * a1112 * z_1**3 * z_2 * q_2 + \
            12 * a1122 * z_1**2 * z_2**2 * q_2 + \
            12 * a1222 * z_1**3 * z_2 * q_2 - \
            6 * a1222 * z_1**2 * z_2**2 * q_2 + \
            5 * a11112 * z_1**5 * q_2 - \
            15 * a11112 * z_1**4 * z_2 * q_2 + \
            10 * a11122 * z_1**4 * z_2 * q_2 - \
            10 * a11122 * z_1**3 * z_2**2 * q_2 + \
            30 * a11222 * z_1**2 * z_2**3 * q_2 - \
            10 * a11222 * z_1**3 * z_2**2 * q_2 + \
            20 * a12222 * z_1**2 * z_2**3 * q_2
    else:
        raise NotImplementedError(f"Wohl order {wohl_order} not supported")

    return ln_gamma_1, ln_gamma_2


def get_wohl_parameters(
        output: torch.Tensor,
        wohl_order: int,
        q_1: torch.Tensor,
        q_2: torch.Tensor,
        molecule_id: int = None,
):
    """
    Get the Wohl coefficients and their names. Order of coefficients must match forward_vle_wohl function.
    """
    if wohl_order == 3:
        coefficients = torch.cat([output,q_1,q_2], axis=1)
        names = ['a12', 'a112', 'a122', 'q1', 'q2']
    elif wohl_order == 4:
        coefficients = torch.cat([output,q_1,q_2], axis=1)
        names = ['a12', 'a112', 'a122', 'a1112', 'a1122', 'a1222', 'q1', 'q2']
    elif wohl_order == 5:
        coefficients = torch.cat([output,q_1,q_2], axis=1)
        names = ['a12', 'a112', 'a122', 'a1112', 'a1122', 'a1222', 'a11112', 'a11122', 'a11222', 'a12222', 'q1', 'q2']
    else:
        raise NotImplementedError(f"Wohl order {wohl_order} not supported")
    if molecule_id is not None:
        names = [f"{name}_{molecule_id}" for name in names]

    return names, coefficients

def forward_vle_nrtl(
    output: torch.Tensor,
    x_1: torch.Tensor,
    x_2: torch.Tensor,
):
    """
    VLE output calculation for the NRTL model
    """
    tau_12, tau_21, alpha = torch.chunk(output, 3, dim=1)
    alpha = torch.sigmoid(alpha)  # Constrain alpha to [0, 1] range
    G_12 = torch.exp(-alpha * tau_12)
    G_21 = torch.exp(-alpha * tau_21)
    
    ln_gamma_1 = x_2**2 * (tau_21 * (G_21 / (x_1 + x_2 * G_21))**2 +
                           tau_12 * G_12 / (x_2 + x_1 * G_12)**2)
    ln_gamma_2 = x_1**2 * (tau_12 * (G_12 / (x_2 + x_1 * G_12))**2 +
                           tau_21 * G_21 / (x_1 + x_2 * G_21)**2)
    
    return ln_gamma_1, ln_gamma_2

def get_nrtl_parameters(
    output: torch.Tensor,
    molecule_id: int = None,
):
    """
    Get the NRTL coefficients and their names.
    """
    names = ['tau_12', 'tau_21', 'alpha']
    # Apply sigmoid to alpha (third column)
    output[:, 2] = torch.sigmoid(output[:, 2])
    if molecule_id is not None:
        names = [f"{name}_{molecule_id}" for name in names]
    return names, output

def forward_vle_nrtl_wohl(
    output: torch.Tensor,
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    q_1: torch.Tensor,
    q_2: torch.Tensor,
    wohl_order: int
):
    """
    VLE output calculation for the NRTL-Wohl hybrid model
    """
    # NRTL calculations
    tau_12, tau_21, alpha = torch.chunk(output[:, :3], 3, dim=1)
    alpha = torch.sigmoid(alpha)  # Constrain alpha to [0, 1] range
    G_12 = torch.exp(-alpha * tau_12)
    G_21 = torch.exp(-alpha * tau_21)
    
    ln_gamma_1_nrtl = x_2**2 * (tau_21 * (G_21 / (x_1 + x_2 * G_21))**2 +
                                tau_12 * G_12 / (x_2 + x_1 * G_12)**2)
    ln_gamma_2_nrtl = x_1**2 * (tau_12 * (G_12 / (x_2 + x_1 * G_12))**2 +
                                tau_21 * G_21 / (x_1 + x_2 * G_21)**2)

    wohl_params = output[:, 3:]
    ln_gamma_1_wohl, ln_gamma_2_wohl = forward_vle_wohl(wohl_params, wohl_order, x_1, x_2, q_1, q_2)

    # Hybrid model
    ln_gamma_1 = ln_gamma_1_nrtl + ln_gamma_1_wohl
    ln_gamma_2 = ln_gamma_2_nrtl + ln_gamma_2_wohl

    return ln_gamma_1, ln_gamma_2

def get_nrtl_wohl_parameters(
    output: torch.Tensor,
    wohl_order: int,
    q_1: torch.Tensor,
    q_2: torch.Tensor,
    molecule_id: int = None,
):
    nrtl_names = ['tau_12', 'tau_21', 'alpha']
    output[:, 2] = torch.sigmoid(output[:, 2])  # Apply sigmoid to alpha
    wohl_names, _ = get_wohl_parameters(output[:, 3:], wohl_order, q_1, q_2)
    names = nrtl_names + wohl_names
    parameters = torch.cat([output, q_1, q_2], axis=1)
    if molecule_id is not None:
        names = [f"{name}_{molecule_id}" for name in names]
    return names, parameters

def forward_vle_uniquac(
    uniquac_params: torch.Tensor,
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    T: torch.Tensor,
    Z: torch.Tensor
):
    """
    VLE output calculation for the UNIQUAC model
    """
    tau12, tau21, r1, r2, q1, q2 = torch.chunk(uniquac_params, 6, dim=1)
    
    # Apply softplus to tau values to ensure they are positive
    tau12 = F.softplus(tau12)
    tau21 = F.softplus(tau21)

  # Ensure r, q are positive and not too small
    r1, r2, q1, q2 = [torch.clamp(param, min=0.1, max=100) for param in [r1, r2, q1, q2]]
    
    # Calculate structural parameters
    l1 = (Z / 2) * (r1 - q1) - (r1 - 1)
    l2 = (Z / 2) * (r2 - q2) - (r2 - 1)
    
    # Calculate area and volume fractions
    theta1 = x_1 * q1 / (x_1 * q1 + x_2 * q2)
    theta2 = x_2 * q2 / (x_1 * q1 + x_2 * q2)
    phi1 = x_1 * r1 / (x_1 * r1 + x_2 * r2)
    phi2 = x_2 * r2 / (x_1 * r1 + x_2 * r2)
    
    # Calculate taus using the correct equation
    R = 8.314  # J/(mol·K)

   # Calculate activity coefficients
    ln_gamma1_c = torch.log(r1 / (x_1 * r1 + x_2 * r2)) - (Z / 2) * q1 * torch.log((r1 * (x_1 * q1 + x_2 * q2)) / ((x_1 * r1 + x_2 * r2) * q1)) + l1 - (r1 / (x_1 * r1 + x_2 * r2)) * x_1 * l1 + x_2 * l2
    ln_gamma1_r = q1 * (1 - torch.log(theta1 + theta2 * tau21) - (theta1 * tau12 / (theta1 + theta2 * tau21)))

    ln_gamma2_c = torch.log(r2 / (x_1 * r1 + x_2 * r2)) + (Z / 2) * q2 * torch.log((r2 * (x_1 * q1 + x_2 * q2)) / ((x_1 * r1 + x_2 * r2) * q2)) + l2 - (r2 / (x_1 * q1 + x_2 * r2)) * x_1 * l1 + x_2 * l2
    ln_gamma2_r = q2 * (1 - torch.log(theta2 + theta1 * tau12) - (theta2 * tau21 / (theta2 + theta1 * tau12)))

    # Clamp ln_gamma values to prevent extreme values
    ln_gamma1 = torch.clamp(ln_gamma1_c + ln_gamma1_r, min=-50, max=50)
    ln_gamma2 = torch.clamp(ln_gamma2_c + ln_gamma2_r, min=-50, max=50)
    
    return ln_gamma1, ln_gamma2

def get_uniquac_parameters(
    output: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    T: torch.Tensor,
    Z: torch.Tensor,
    molecule_id: int = None
):
    tau12, tau21 = torch.chunk(output, 2, dim=1)
    # Apply softplus to tau values
    tau12 = F.softplus(tau12)
    tau21 = F.softplus(tau21)
    # Ensure r and q values are positive and not too small
    r1 = torch.clamp(r1, min=0.1, max=100)
    r2 = torch.clamp(r2, min=0.1, max=100)
    q1 = torch.clamp(q1, min=0.1, max=100)
    q2 = torch.clamp(q2, min=0.1, max=100)

    # Calculate structural parameters
    l1 = (Z / 2) * (r1 - q1) - (r1 - 1)
    l2 = (Z / 2) * (r2 - q2) - (r2 - 1)

    # Calculate area and volume fractions
    theta1 = x_1 * q1 / (x_1 * q1 + x_2 * q2)
    theta2 = x_2 * q2 / (x_1 * q1 + x_2 * q2)
    phi1 = x_1 * r1 / (x_1 * r1 + x_2 * r2)
    phi2 = x_2 * r2 / (x_1 * r1 + x_2 * r2)

    # Calculate taus using the delta_u values
    R = 8.314  # J/(mol·K)
    

    names = ['tau12', 'tau21', 'r1', 'r2', 'q1', 'q2', 'theta1', 'theta2', 'phi1', 'phi2', 'l1', 'l2', 'Z']
    parameters = torch.cat([tau12, tau21, r1, r2, q1, q2, theta1, theta2, phi1, phi2, l1, l2, Z], dim=1)

    if molecule_id is not None:
        names = [f"{name}_{molecule_id}" for name in names]

    return names, parameters

def forward_vle_freestyle(
    output: torch.Tensor,  # size(batch, 1)
    features: torch.Tensor, # size(batch, 3) x1, x2, T
):
    """
    VLE output using a non-templated freestyle model.
    """
    gE = output  # size(batch)
    x1 = features[:, [0]]  # size(batch)
    x2 = features[:, [1]]  # size(batch)
    T = features[:, [2]]  # size(batch)

    dgE = torch.autograd.grad(gE.sum(), features, create_graph=True)[0]  # size(batch, 3)

    dgE_dx1 = dgE[:, [0]]  # size(batch)
    dgE_dx2 = dgE[:, [1]]  # size(batch)

    ln_gamma_1 = (gE + dgE_dx1 * x2 - dgE_dx2 * x2) / T
    ln_gamma_2 = (gE + dgE_dx2 * x1 - dgE_dx1 * x1) / T


    return ln_gamma_1, ln_gamma_2

def unscale_vle_parameters(
        parameters: np.ndarray,
        target_scaler,
        hybrid_model_features_scaler,
        vle: str,
):
    # none of the current methods are affected by scaling
    return parameters