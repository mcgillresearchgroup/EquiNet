# EquiNet: Predicting Vapor-Liquid Equilibrium with Physics-Informed Neural Networks

EquiNet is a deep learning framework based on the Chemprop architecture, designed for predicting vapor-liquid equilibrium (VLE) properties of binary mixtures. It incorporates physicochemical constraints using physics-informed neural networks (PINNs) to enhance thermodynamic consistency and accuracy.

---

## Installation & Setup Instructions

### Option 1: Quick Install (via pip)

If you just want to use EquiNet without editing the source, install it directly from PyPI:

```bash
pip install equinet
```

This installs the `equinet` package along with the `equinet-train`, `equinet-predict`, `equinet-parameters`, and `equinet-hyperopt` command-line tools (see [Running EquiNet](#5-running-equinet) below). For development, or if you prefer managing dependencies (like RDKit and PyTorch) via conda, follow the steps below instead.

### Option 2: Anaconda Install from Source

Use one of several available methods ([GitHub documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository))to clone this repository.
```bash
git clone https://github.com/mcgillresearchgroup/equinet.git
```

If using Windows, open the Anaconda Prompt (not a regular terminal).

Navigate to the project directory:

```bash 
cd path/to/equinet
```

Create the environment. The mamba solver is recommended but not required.

```bash
conda env create -f environment.yml --solver=libmamba
```

Activate the environment:

```bash
conda activate equinet
```
Complete EquiNet setup locally:

```bash
pip install -e .

```
Note the trailing `.` in the command, it is important. This also installs the `equinet-train`, `equinet-predict`, `equinet-parameters`, and `equinet-hyperopt` command-line tools.

## Dataset Preparation
The data needs to be split into two .csv files, a **targets** file and a **features** file.

For the Targets File (in training):
- The **targets file** must contain columns in the following order: 'SMILE 1', 'SMILE 2', 'y1', 'y2', 'log10P', 'lngamma1', 'lngamma2', 'log10P1sat', 'log10P2sat'.
- If targets are not needed for all these options, the number of columns can be truncated, but columns in the middle cannot be skipped. Typical training associated with the paper involves these columns: 'SMILE 1', 'SMILE 2', 'y1', 'y2', 'log10P'.
- All targets are not needed for training. If individual targets are not known, they should be left blank in the csv.
- `SMILE 1` and `SMILE 2` are the SMILES representations of the two components in the binary mixture. `SMILE 1` and `SMILE 2` should be valid **RDKit-compliant SMILES** strings.
- `y1` and `y2` are the mole fractions of components 1 and 2, respectively, and must be in the range [0, 1]. They must sum to 1.
- `log10P` is the logarithm (base 10) of the total pressure in **Pascals (Pa)**.
- `lngamma1` and `lngamma2` are the natural logarithm (base e) of the activity coefficients.
- `log10P1sat` and `log10P2sat` are the logarithm (base 10) of the component vapor pressures and must be in **Pascals (Pa)**.

For the Targets File (in prediction):
- During prediction, the **targets file** must contain columns 'SMILE 1' and 'SMILE 2' in the first two columns. No other columns are necessary and will be ignored.
- `SMILE 1` and `SMILE 2` are the SMILES representations of the two components in the binary mixture. `SMILE 1` and `SMILE 2` should be valid **RDKit-compliant SMILES** strings.

For the Features File (both training and prediction):
- The **features file** must contain the following columns: 'x1', 'x2', 'T(K)', 'log10P1sat', 'log10P2sat'
- Unlike the targets tile, none of these values can be left blank or columns omitted.
- `x1` and `x2` are the mole fractions of components 1 and 2, respectively, and must be in the range [0, 1]. They must sum to 1.
- `T(K)` is the temperature in **Kelvin**.
- `log10P1sat` and `log10P2sat` are the base-10 logarithms of the **pure component saturation pressures**, also in **Pascals (Pa)**.
- If internal vapor pressure prediction is being used, then the provided values for `log10P1sat` and `log10P2sat` will not be referenced. They do still have to be provided and can be filled with `nan` as their value if desired.

Ensure both files are aligned row-wise and contain corresponding data points for training or prediction and are CSV files.

## Running EquiNet
### 🧪 Training & Prediction on HPC (Bash Script Setup)

To run training and prediction jobs, a typical `bash` script looks like the following:

```bash
data_dir= \yourpath\to\data
results_dir=\yourpath\to\results
equinet_path=\yourpath\to\equinet

python $equinet_path/train.py \
  --data_path $data_dir/targets.csv \
  --features_path $data_dir/features.csv \
  --dataset_type regression \
  --epochs 30 \
  --save_dir $results_dir \
  --split_type random_binary_pairs \
  --vle activity \
  --vp antoine \
  --binary_equivariant \
  --self_activity_correction \
  --config_path config.json \
  --aggregation norm \
  --save_smiles_splits

python $equinet_path/predict.py \
  --test_path $results_dir/fold_0/test_full.csv \
  --features_path $results_dir/fold_0/test_features.csv \
  --preds_path $results_dir/test_preds.csv \
  --checkpoint_dir $results_dir \
  --number_of_molecules 2 \
  --drop_extra_columns

python $equinet_path/parameters.py \
  --test_path $results_dir/fold_0/test_full.csv \
  --features_path $results_dir/fold_0/test_features.csv \
  --preds_path $results_dir/test_params.csv \
  --checkpoint_dir $results_dir \
  --number_of_molecules 2 \
  --drop_extra_columns
```

If you installed EquiNet via pip, the same commands are available as `equinet-train`, `equinet-predict`, and `equinet-parameters` console commands, taking the same arguments (no need to reference `$equinet_path`).

### Switching Between Model Types
EquiNet supports multiple model types for VLE prediction via the --vle and --vp flags:

--vle sets the activity coefficient model. Options include:
- basic – no thermodynamic constraints
- activity – activity-based PINN model
- nrtl – Non-Random Two-Liquid model
- nrtl-wohl – NRTL with Wohl interaction form
- wohl – full Wohl expansion (3rd–5th order depending on config)

--wohl_order – Wohl expansion with specified order (e.g., 3, 4, or 5) for the Wohl expansion, if Wohl or NRTL-Wohl methods are used.

--vp sets the vapor pressure prediction method:
- Leave empty (omit --vp) → tabulated vapor pressure from features file is used
- Set --vp antoine → model internally predicts vapor pressure using Antoine equation

## API Inference Functions

For quick, in-process predictions without writing intermediate CSV files or using the CLI, `equinet.inference` exposes a set of Python functions that wrap a pretrained model. Each function accepts SMILES strings for the two components and an optional `model_path` (defaults to a packaged pretrained checkpoint).

```python
from equinet.inference import (
    predict_vle_single_point,
    predict_vle_isothermal_envelope,
    predict_vle_isobaric_envelope,
    predict_vle_parameters,
)
```

- `predict_vle_single_point` – Predicts activity coefficients, vapor compositions, and vapor pressures for a single composition (`x1`, `x2`) at a given temperature.
- `predict_vle_isothermal_envelope` – Predicts a full VLE envelope at a fixed temperature over a mesh of `x1` compositions.
- `predict_vle_isobaric_envelope` – Predicts a full VLE envelope at a fixed pressure, solving internally for the temperature that matches the target pressure at each composition.
- `predict_vle_parameters` – Predicts the underlying thermodynamic model parameters (e.g. NRTL `tau`/`alpha` and Antoine coefficients) for a binary mixture, rather than pointwise VLE properties.

See the notebooks in [examples/](examples/) for usage examples.
