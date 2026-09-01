import numpy as np
from scipy.optimize import least_squares

from equinet.train import make_predictions, load_model, get_parameters
from equinet.args import PredictArgs, ParameterArgs
from equinet.inference.utils import NoPrint, create_temp_csv_paths, write_vle_input_files, build_predict_arguments
from equinet.inference.weights import get_pretrained_model_path


def predict_vp(
        smiles: str,
        temperature: float | list[float] | np.ndarray,
        model_path: str = None,
    ) -> np.ndarray:
    """Predict a component vapor pressure from its SMILES string and temperature.

    Args:
        smiles: SMILES string of the component.
        temperature: Temperature in Kelvin. Can be provided as a float or a list or numpy array of floats.
        model_path: Path to a model checkpoint. Defaults to the packaged pretrained model.

    Returns:
        A numpy array of predicted vapor pressures.
    """

    if model_path is None:
        model_path = get_pretrained_model_path("equinet_v0.2.0.pt")

    if len(np.array(temperature).shape) == 0:
        temperature = [temperature]

    temp_test, temp_features, temp_preds = create_temp_csv_paths()
    write_vle_input_files(temp_test, temp_features, smiles, smiles, [0.5]*len(temperature), [0.5]*len(temperature), temperature)

    args = PredictArgs().parse_args(
        build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
    )

    with NoPrint():
        # load pretrained model
        model_objects = load_model(args=args)

        # make predictions
        preds = make_predictions(args=args, model_objects=model_objects) # y1, y2, log10P, lngamma1, lngamma2, log10P1sat, log10P2sat

    preds = np.array(preds)
    log10P1sat = preds[:, 5]
    P1sat = 10 ** log10P1sat
    return P1sat


def predict_bp(
        smiles: str,
        pressure: float = 101325, # Pa
        model_path: str = None,
    ) -> float:
    """Predict a component boiling point from its SMILES string.

    Args:
        smiles: SMILES string of the component.
        pressure: The pressure in Pascals at which to predict the boiling point. Defaults to 101325 Pa (1 atm).
        model_path: Path to a model checkpoint. Defaults to the packaged pretrained model.
    
    Returns:
        The predicted boiling point in Kelvin.
    """

    if model_path is None:
        model_path = get_pretrained_model_path("equinet_v0.2.0.pt")
    logP_target = np.log10(pressure)

    temp_test, temp_features, temp_preds = create_temp_csv_paths()

    temperature = np.linspace(100, 800, 701)
    write_vle_input_files(temp_test, temp_features, smiles, smiles, [0.5]*len(temperature), [0.5]*len(temperature), temperature)

    args = PredictArgs().parse_args(
        build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
    )

    with NoPrint():
        # load pretrained model
        model_objects = load_model(args=args)

        # make predictions
        preds = make_predictions(args=args, model_objects=model_objects)

    # Finding T closest to the target pressure
    preds = np.array(preds)
    logP1sats = preds[:, 5]
    closest_P1_index = np.argmin(np.abs(logP1sats - logP_target))
    closest_P1_T = temperature[closest_P1_index]

    # create a solver objective function
    def predict_P_objective(T):
        write_vle_input_files(temp_test, temp_features, smiles, smiles, 0.5, 0.5, T)

        args = PredictArgs().parse_args(
            build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
        )

        with NoPrint():
            preds = make_predictions(args=args, model_objects=model_objects)
            P_preds = 10 ** np.array(preds)[:, 5]

        return P_preds - pressure


    T_solution = least_squares(
        predict_P_objective,
        closest_P1_T,
    ).x

    return T_solution


def predict_vp_parameters(
        smiles: str,
        temperature: float,
        model_path: str = None,
    ):
    """Predict the modified Antoine's parameters for a component at a given temperature.

    log10P = a - softplus(b) / softplus(T / t_scale + c, beta=0.1), where softplus(x, beta) = (1/beta) * log(1 + exp(beta * x)).

    Args:
        smiles: SMILES string of the component.
        temperature: Temperature in Kelvin.
        model_path: Path to a model checkpoint. Defaults to the packaged pretrained model (the same one used by predict_vp).

    Returns:
        A dict mapping parameter names ('antoine_a', 'antoine_b', 'antoine_c', 'antoine_t_scale') to their predicted values.
    """

    if model_path is None:
        model_path = get_pretrained_model_path("equinet_v0.2.0.pt")

    temp_test, temp_features, temp_preds = create_temp_csv_paths()
    write_vle_input_files(temp_test, temp_features, smiles, smiles, 0.5, 0.5, temperature)

    args = ParameterArgs().parse_args(
        build_predict_arguments(temp_test, temp_features, temp_preds, model_path)
    )

    with NoPrint():
        names, parameters = get_parameters(args=args)

    preds_dict = dict(zip(names, parameters[0]))

    return_dict = {
        'antoine_a': preds_dict['antoine_a_1'],
        'antoine_b': preds_dict['antoine_b_1'],
        'antoine_c': preds_dict['antoine_c_1'],
        'antoine_t_scale': preds_dict['antoine_t_scale_1'],
    }

    return return_dict