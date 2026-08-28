import numpy as np
from scipy.optimize import least_squares

from equinet.train import make_predictions, load_model, get_parameters
from equinet.args import PredictArgs, ParameterArgs
from equinet.inference.utils import NoPrint, create_temp_csv_paths, write_vle_input_files, build_predict_arguments
from equinet.inference.weights import get_pretrained_model_path


def predict_vle_single_point(
        smiles_1: str,
        smiles_2: str,
        x1: float, # within [0, 1]
        x2: float, # within [0, 1]
        temperature: float,
        model_path: str = None,
    ):
    """Predict VLE activity coefficients and vapor pressures for a single (x1, x2, T) point.

    Args:
        smiles_1: SMILES string of the first component.
        smiles_2: SMILES string of the second component.
        x1: Liquid mole fraction of component 1, within [0, 1].
        x2: Liquid mole fraction of component 2, within [0, 1].
        temperature: Temperature in Kelvin.
        model_path: Path to a model checkpoint. Defaults to the packaged pretrained model.

    Returns:
        A dict with keys 'x1', 'x2', 'T', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat'.
    """

    if model_path is None:
        model_path = get_pretrained_model_path("equinet_v0.2.0.pt")

    temp_test, temp_features, temp_preds = create_temp_csv_paths()
    write_vle_input_files(temp_test, temp_features, smiles_1, smiles_2, x1, x2, temperature)

    args = PredictArgs().parse_args(
        build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
    )

    with NoPrint():
        # load pretrained model
        model_objects = load_model(args=args)

        # make predictions
        preds = make_predictions(args=args, model_objects=model_objects)

    preds_dict = dict(zip(['y1', 'y2', 'log10P', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat'], preds[0]))
    preds_dict['x1'] = x1
    preds_dict['x2'] = x2
    preds_dict['T'] = temperature
    return preds_dict


def predict_vle_isothermal_envelope(
        smiles_1: str,
        smiles_2: str,
        temperature: float,
        mesh_size: int = 101,
        model_path: str = None,
    ):
    """Predict a VLE envelope at fixed temperature over a mesh of x1 compositions.

    Args:
        smiles_1: SMILES string of the first component.
        smiles_2: SMILES string of the second component.
        temperature: Temperature in Kelvin, held fixed across the envelope.
        mesh_size: Number of x1 points spanning [0, 1].
        model_path: Path to a model checkpoint. Defaults to the packaged pretrained model.

    Returns:
        A dict with keys 'x1', 'x2', 'T', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat',
        each mapped to an array of length mesh_size.
    """

    if model_path is None:
        model_path = get_pretrained_model_path("equinet_v0.2.0.pt")

    x1s = np.linspace(0, 1, mesh_size)
    x2s = 1 - x1s

    temp_test, temp_features, temp_preds = create_temp_csv_paths()
    write_vle_input_files(temp_test, temp_features, smiles_1, smiles_2, x1s, x2s, temperature)

    args = PredictArgs().parse_args(
        build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
    )

    with NoPrint():
        # load pretrained model
        model_objects = load_model(args=args)

        # make predictions
        preds = make_predictions(args=args, model_objects=model_objects)

    preds = np.array(preds)

    # make a dictionary of the predictions split by columns
    preds_dict = dict(zip(['y1', 'y2', 'log10P', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat'], preds.T))
    preds_dict['x1'] = x1s
    preds_dict['x2'] = x2s
    preds_dict['T'] = np.full(mesh_size, temperature)

    return preds_dict


def predict_vle_isobaric_envelope(
        smiles_1: str,
        smiles_2: str,
        pressure: float = 101325,
        mesh_size: int = 101,
        model_path: str = None,
    ):
    """Predict a VLE envelope at fixed pressure, solving for the T that matches the target pressure at each x1.

    Args:
        smiles_1: SMILES string of the first component.
        smiles_2: SMILES string of the second component.
        pressure: Target pressure in Pa, held fixed across the envelope.
        mesh_size: Number of x1 points spanning [0, 1].
        model_path: Path to a model checkpoint. Defaults to the packaged pretrained model.

    Returns:
        A dict with keys 'x1', 'x2', 'T', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat',
        each mapped to an array of length mesh_size.
    """

    if model_path is None:
        model_path = get_pretrained_model_path("equinet_v0.2.0.pt")

    logP_target = np.log10(pressure)

    x1s = np.linspace(0, 1, mesh_size)
    x2s = 1 - x1s

    temp_test, temp_features, temp_preds = create_temp_csv_paths()

    # first, calculating vapor pressures of the two components at a range of
    # temperatures to find initial guesses for the isobaric envelope.

    Ts = np.linspace(100, 800, 701)
    write_vle_input_files(temp_test, temp_features, smiles_1, smiles_2, 0.5, 0.5, Ts)

    args = PredictArgs().parse_args(
        build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
    )

    with NoPrint():
        # load pretrained model
        model_objects = load_model(args=args)

        # make predictions
        preds = make_predictions(args=args, model_objects=model_objects)

    preds = np.array(preds)

    # Finding T closest to the target pressure for each component
    logP1sats = preds[:, 5]
    logP2sats = preds[:, 6]

    closest_P1_index = np.argmin(np.abs(logP1sats - logP_target))
    closest_P2_index = np.argmin(np.abs(logP2sats - logP_target))

    closest_P1_T = Ts[closest_P1_index]
    closest_P2_T = Ts[closest_P2_index]
    T_guess = np.linspace(closest_P2_T, closest_P1_T, mesh_size)

    # create a solver objective function
    def predict_P_objective(T):
        write_vle_input_files(temp_test, temp_features, smiles_1, smiles_2, x1s, x2s, T)

        args = PredictArgs().parse_args(
            build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
        )

        with NoPrint():
            preds = make_predictions(args=args, model_objects=model_objects)
            P_preds = 10 ** np.array(preds)[:, 2]

        return P_preds - pressure

    T_solution = least_squares(
        predict_P_objective,
        T_guess,
        diff_step=1e-4,
        jac_sparsity=np.identity(mesh_size)
    ).x

    # re-predict with the final T to get the final y
    write_vle_input_files(temp_test, temp_features, smiles_1, smiles_2, x1s, x2s, T_solution)

    args = PredictArgs().parse_args(
        build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
    )

    with NoPrint():
        preds = np.array(make_predictions(args=args, model_objects=model_objects))

    # make a dictionary of the predictions split by columns
    preds_dict = dict(zip(['y1', 'y2', 'log10P', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat'], preds.T))
    preds_dict['x1'] = x1s
    preds_dict['x2'] = x2s
    preds_dict['T'] = T_solution

    return preds_dict


def predict_vle_parameters(
        smiles_1: str,
        smiles_2: str,
        temperature: float,
        x1: float = 0.5, # within [0, 1]
        x2: float = 0.5, # within [0, 1]
        model_path: str = None,
    ):
    """Predict the NRTL/Antoine parameters (tau, alpha, Antoine coefficients) for a binary mixture.

    Args:
        smiles_1: SMILES string of the first component.
        smiles_2: SMILES string of the second component.
        temperature: Temperature in Kelvin.
        x1: Liquid mole fraction of component 1, within [0, 1].
        x2: Liquid mole fraction of component 2, within [0, 1].
        model_path: Path to a model checkpoint. Defaults to the packaged
            no-self-activity-correction pretrained model.

    Returns:
        A dict mapping parameter names (e.g. 'tau_12', 'tau_21', 'alpha',
        'antoine_a_1', 'antoine_b_1', 'antoine_c_1', 'antoine_a_2', 'antoine_b_2',
        'antoine_c_2') to their predicted values.
    """

    if model_path is None:
        model_path = get_pretrained_model_path("equinet_no-self-activity-correction_v0.2.0.pt")

    temp_test, temp_features, temp_preds = create_temp_csv_paths()
    write_vle_input_files(temp_test, temp_features, smiles_1, smiles_2, x1, x2, temperature)

    args = ParameterArgs().parse_args(
        build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
    )

    with NoPrint():
        names, parameters = get_parameters(args=args)

    preds_dict = dict(zip(names, parameters[0]))

    return preds_dict