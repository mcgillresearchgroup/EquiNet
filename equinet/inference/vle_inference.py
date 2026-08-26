import pandas as pd
import numpy as np
import tempfile
from importlib.resources import files, as_file
from scipy.optimize import least_squares

from equinet.train import make_predictions, load_model
from equinet.args import PredictArgs
from equinet.utils import NoPrint

def predict_vle_single_point(
        smiles_1: str,
        smiles_2: str,
        x1: float, # within [0, 1]
        x2: float, # within [0, 1]
        temperature: float,
        model_path: str = None,
    ):

    if model_path is None:
        with as_file(files("equinet").joinpath("pretrained_models", "equinet_v0.2.0.pt")) as p:
            model_path = str(p)

    # create a dummy input file
    f_preds, temp_preds = tempfile.mkstemp(suffix='.csv')
    f_features, temp_features = tempfile.mkstemp(suffix='.csv')
    f_test, temp_test = tempfile.mkstemp(suffix='.csv')

    test_df = pd.DataFrame({
        "smiles_1": [smiles_1],
        "smiles_2": [smiles_2]
    })
    features_df = pd.DataFrame({
        'x1': [x1],
        'x2': [x2],
        'T': [temperature],
        'log10P1sat': ["nan"],
        'log10P2sat': ["nan"],
    })

    # store the input dataframes
    test_df.to_csv(temp_test, index=False)
    features_df.to_csv(temp_features, index=False)

    # provide and parse arguments
    arguments = [
        '--test_path', temp_test,
        '--features_path', temp_features,
        '--preds_path', temp_preds,
        '--checkpoint_path', model_path,
        '--number_of_molecules', '2',
        '--num_workers', '0',
    ]
    args = PredictArgs().parse_args(arguments)

    with NoPrint():
        # load pretrained model
        model_objects = load_model(args=args)

        # make predictions
        preds = make_predictions(args=args, model_objects=model_objects)

    preds_dict = dict(zip(['x1', 'x2', 'T', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat'], preds[0]))

    return preds_dict


def predict_vle_isothermal_envelope(
        smiles_1: str,
        smiles_2: str,
        temperature: float,
        mesh_size: int = 101,
        model_path: str = None,
    ):

    if model_path is None:
        with as_file(files("equinet").joinpath("pretrained_models", "equinet_v0.2.0.pt")) as p:
            model_path = str(p)

    # create a dummy input file
    f_preds, temp_preds = tempfile.mkstemp(suffix='.csv')
    f_features, temp_features = tempfile.mkstemp(suffix='.csv')
    f_test, temp_test = tempfile.mkstemp(suffix='.csv')

    x1s = np.linspace(0, 1, mesh_size)
    x2s = 1 - x1s

    test_df = pd.DataFrame({
        "smiles_1": [smiles_1] * mesh_size,
        "smiles_2": [smiles_2] * mesh_size
    })
    features_df = pd.DataFrame({
        'x1': x1s,
        'x2': x2s,
        'T': [temperature] * mesh_size,
        'log10P1sat': ["nan"] * mesh_size,
        'log10P2sat': ["nan"] * mesh_size,
    })

    # store the input dataframes
    test_df.to_csv(temp_test, index=False)
    features_df.to_csv(temp_features, index=False)

    # provide and parse arguments
    arguments = [
        '--test_path', temp_test,
        '--features_path', temp_features,
        '--preds_path', temp_preds,
        '--checkpoint_path', model_path,
        '--number_of_molecules', '2',
        '--num_workers', '0',
    ]
    args = PredictArgs().parse_args(arguments)

    with NoPrint():
        # load pretrained model
        model_objects = load_model(args=args)

        # make predictions
        preds = make_predictions(args=args, model_objects=model_objects)

    preds = np.array(preds)

    # make a dictionary of the predictions split by columns
    preds_dict = dict(zip(['x1', 'x2', 'T', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat'], preds.T))

    return preds_dict


def predict_vle_isobaric_envelope(
        smiles_1: str,
        smiles_2: str,
        pressure: float = 101325,
        mesh_size: int = 101,
        model_path: str = None,
    ):

    if model_path is None:
        with as_file(files("equinet").joinpath("pretrained_models", "equinet_v0.2.0.pt")) as p:
            model_path = str(p)

    logP_target = np.log10(pressure)

    x1s = np.linspace(0, 1, mesh_size)
    x2s = 1 - x1s

    # create a dummy input file
    f_preds, temp_preds = tempfile.mkstemp(suffix='.csv')
    f_features, temp_features = tempfile.mkstemp(suffix='.csv')
    f_test, temp_test = tempfile.mkstemp(suffix='.csv')

    # first, calculating vapor pressures of the two components at a range of
    # temperatures to find initial guesses for the isobaric envelope.

    Ts = np.linspace(100, 800, 701)

    test_df = pd.DataFrame({
        "smiles_1": [smiles_1] * 701,
        "smiles_2": [smiles_2] * 701
    })
    features_df = pd.DataFrame({
        'x1': [0.5] * 701,
        'x2': [0.5] * 701,
        'T': Ts,
        'log10P1sat': ["nan"] * 701,
        'log10P2sat': ["nan"] * 701,
    })

    # store the input dataframes
    test_df.to_csv(temp_test, index=False)
    features_df.to_csv(temp_features, index=False)

    # provide and parse arguments
    arguments = [
        '--test_path', temp_test,
        '--features_path', temp_features,
        '--preds_path', temp_preds,
        '--checkpoint_path', model_path,
        '--number_of_molecules', '2',
        '--num_workers', '0',
    ]
    args = PredictArgs().parse_args(arguments)

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
        test_df = pd.DataFrame({
            "smiles_1": [smiles_1] * mesh_size,
            "smiles_2": [smiles_2] * mesh_size
        })
        features_df = pd.DataFrame({
            'x1': x1s,
            'x2': x2s,
            'T': T,
            'log10P1sat': ["nan"] * mesh_size,
            'log10P2sat': ["nan"] * mesh_size,
        })

        # store the input dataframes
        test_df.to_csv(temp_test, index=False)
        features_df.to_csv(temp_features, index=False)

        # provide and parse arguments
        arguments = [
            '--test_path', temp_test,
            '--features_path', temp_features,
            '--preds_path', temp_preds,
            '--checkpoint_path', model_path,
            '--number_of_molecules', '2',
            '--num_workers', '0',
        ]
        args = PredictArgs().parse_args(arguments)

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

    test_df = pd.DataFrame({
        'smiles1': [smiles_1]*mesh_size,
        'smiles2': [smiles_2]*mesh_size,
    })
    features_df = pd.DataFrame({
        'x1': np.linspace(0, 1, mesh_size),
        'x2': np.linspace(1, 0, mesh_size),
        'T': T_solution, # T is a list of temperatures 
        'log10P1sat': ["nan"]*mesh_size,
        'log10P2sat': ["nan"]*mesh_size,
    })
    test_df.to_csv(temp_test, index=False)
    features_df.to_csv(temp_features, index=False)

    args = PredictArgs().parse_args([
        '--test_path', temp_test,
        '--features_path', temp_features,
        '--preds_path', temp_preds,
        '--checkpoint_path', model_path,
        '--number_of_molecules', '2',
        '--num_workers', '0',
    ])

    with NoPrint():
        preds = np.array(make_predictions(args=args, model_objects=model_objects)) 

    # make a dictionary of the predictions split by columns
    preds_dict = dict(zip(['x1', 'x2', 'T', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat'], preds.T))

    return preds_dict
