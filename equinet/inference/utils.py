import sys
import tempfile
from io import StringIO
from importlib.resources import files, as_file

import numpy as np
import pandas as pd


class NoPrint:
    """
    Context manager to suppress print statements.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_sterr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_sterr


def create_temp_csv_paths():
    """Create temp CSV paths used for the test, features, and predictions files."""
    _, temp_test = tempfile.mkstemp(suffix='.csv')
    _, temp_features = tempfile.mkstemp(suffix='.csv')
    _, temp_preds = tempfile.mkstemp(suffix='.csv')
    return temp_test, temp_features, temp_preds


def write_vle_input_files(
        temp_test: str,
        temp_features: str,
        smiles_1: str,
        smiles_2: str,
        x1,
        x2,
        T,
        log10P1sat="nan",
        log10P2sat="nan",
    ):
    """Write the test/features CSVs for a VLE prediction, broadcasting scalars to match x1/x2/T length."""
    n = max(np.size(x1), np.size(x2), np.size(T))

    def _broadcast(value):
        return [value] * n if np.ndim(value) == 0 else value

    test_df = pd.DataFrame({
        "smiles_1": [smiles_1] * n,
        "smiles_2": [smiles_2] * n,
    })
    features_df = pd.DataFrame({
        'x1': _broadcast(x1),
        'x2': _broadcast(x2),
        'T': _broadcast(T),
        'log10P1sat': _broadcast(log10P1sat),
        'log10P2sat': _broadcast(log10P2sat),
    })

    test_df.to_csv(temp_test, index=False)
    features_df.to_csv(temp_features, index=False)


def build_predict_arguments(temp_test: str, temp_features: str, temp_preds: str, model_path: str):
    """Build the CLI-style argument list shared by prediction/parameter calls."""
    return [
        '--test_path', temp_test,
        '--features_path', temp_features,
        '--preds_path', temp_preds,
        '--checkpoint_path', model_path,
        '--number_of_molecules', '2',
        '--num_workers', '0',
    ]