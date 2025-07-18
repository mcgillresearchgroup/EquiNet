# EquiNet: Predicting Vapor-Liquid Equilibrium with Physics-Informed Neural Networks

EquiNet is a deep learning framework based on the Chemprop architecture, designed for predicting vapor-liquid equilibrium (VLE) properties of binary mixtures. It incorporates physicochemical constraints using physics-informed neural networks (PINNs) to enhance thermodynamic consistency and accuracy.

---

## Installation & Setup Instructions

### 1. Install Anaconda

Download and install [Anaconda](https://www.anaconda.com/).  
If using a Mac with an M1/M2 chip, go to “Other Installers” to choose the correct architecture.

### 2. Clone This Repository

Use one of several available methods ([GitHub documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository))to clone this repository.
```bash
git clone https://github.com/mcgillresearchgroup/equinet.git
```
### 3. Create a New Conda Environment
If using Windows, open the Anaconda Prompt (not a regular terminal).

Navigate to the project directory:

```bash 
cd path/to/equinet
```

Create the environment:

```bash
conda env create -f environment.yml
```
If the solution is slow or hangs, you can use the mamba solver. This is the default on newer Anaconda installations, but has to be manually selected on older ones:

```bash
conda env create -f environment.yml --solver=libmamba
```
Activate the environment:

```bash
conda activate chemprop
```
Complete EquiNet setup locally:

```bash
pip install -e .

```
Note the trailing `.` in the command, it is important.

### 4. Dataset Preparation
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

### 5. Running EquiNet
## 🧪 Training & Prediction on HPC (Bash Script Setup)

To run training and prediction on an HPC server with SLURM, a typical `bash` script looks like the following:

```bash
data_dir= \yourpath\to\data
results_dir=\yourpath\to\results
chemprop_path=\yourpath\to\chemprop

python $chemprop_path/train.py \
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

python $chemprop_path/predict.py \
  --test_path $results_dir/fold_0/test_full.csv \
  --features_path $results_dir/fold_0/test_features.csv \
  --preds_path $results_dir/test_preds.csv \
  --checkpoint_dir $results_dir \
  --number_of_molecules 2 \
  --drop_extra_columns

python $chemprop_path/parameters.py \
  --test_path $results_dir/fold_0/test_full.csv \
  --features_path $results_dir/fold_0/test_features.csv \
  --preds_path $results_dir/test_params.csv \
  --checkpoint_dir $results_dir \
  --number_of_molecules 2 \
  --drop_extra_columns
```
#### Switching Between Model Types
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

