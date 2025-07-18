from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from .ffn import build_ffn, binary_equivariant_readout
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import initialize_weights
from .vp import forward_vp, get_vp_parameter_names
from .vle import forward_vle_basic, forward_vle_activity, forward_vle_wohl, forward_vle_nrtl, forward_vle_nrtl_wohl, get_wohl_parameters, get_nrtl_parameters, get_nrtl_wohl_parameters, forward_vle_uniquac, get_uniquac_parameters, get_activity_parameters, forward_vle_freestyle

class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()
        self.args = args
        self.is_atom_bond_targets = False
        self.classification = args.dataset_type == "classification"
        self.multiclass = args.dataset_type == "multiclass"
        self.loss_function = args.loss_function
        self.vp = args.vp
        self.vle = args.vle
        self.wohl_order = args.wohl_order
        self.fugacity_balance = args.fugacity_balance
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.noisy_temperature = args.noisy_temperature
        self.sigmoid = nn.functional.sigmoid
        self.softplus = nn.functional.softplus
        self.binary_equivariant = args.binary_equivariant
        self.self_activity_correction = args.self_activity_correction
        self.self_activity_lambda = args.self_activity_lambda
        self.uniquac_z = args.uniquac_z
        self.learn_uniquac_z = args.learn_uniquac_z
        self.output_size = args.num_tasks

        if self.vp is not None:
            vp_number_parameters_dict = {"basic": 1,"antoine": 3}
            self.vp_output_size = vp_number_parameters_dict[self.vp]
            if self.vle is None:
                self.output_size = self.vp_output_size

        if self.vle is not None:
            if self.vle == "basic":
                self.vle_output_size = 7 # y_1, y_2, log10P, ln_gamma_1, ln_gamma_2, log10p1sat, log10p2sat
            elif self.vle == "activity":
                self.vle_output_size = 2 # gamma_1, gamma_2
            elif self.vle == "wohl":
                wohl_number_parameters_dict = {2: 1, 3: 3, 4: 6, 5: 10}
                self.vle_output_size = wohl_number_parameters_dict[self.wohl_order]
            elif self.vle == "nrtl-wohl":
                wohl_number_parameters_dict = {3: 3, 4: 6, 5: 10}
                self.vle_output_size = 3 + wohl_number_parameters_dict[self.wohl_order]  # NRTL params + Wohl params
            elif self.vle == "nrtl":
                self.vle_output_size = 3 # tau12 tau21 alpha
            elif self.vle == "uniquac":
                self.vle_output_size = 2  # tau12, tau21
            elif self.vle == "freestyle":
                self.vle_output_size = 1  # gE
            self.output_size = self.vle_output_size
            

        if self.binary_equivariant:
            if self.vle == "uniquac":
                self.output_equivariant_pairs = [(0,1), (2,3), (4,5), (6,7)]  # u12-u21, u11-u22, r1-r2, q1-q2
                self.features_equivariant_pairs = []  # T
            elif self.vle == "wohl":
                if self.wohl_order == 2: # a12; a112, a122; a1112, a1122, a1222; a11112, a11122, a11222, a12222
                    self.output_equivariant_pairs = []
                    self.features_equivariant_pairs = [] # T
                elif self.wohl_order == 3:
                    self.output_equivariant_pairs = [(1,2)]
                    self.features_equivariant_pairs = []
                elif self.wohl_order == 4:
                    self.output_equivariant_pairs = [(1,2), (3,5)]
                    self.features_equivariant_pairs = []
                elif self.wohl_order == 5:
                    self.output_equivariant_pairs = [(1,2), (3,5), (6,9), (7,8)]
                    self.features_equivariant_pairs = []
                else:
                    raise ValueError(f"Unsupported equivariant method with wohl order {self.wohl_order}.")
            elif self.vle == "nrtl":
                self.output_equivariant_pairs = [(0,1)] # tau_12, tau_21 alpha
                self.features_equivariant_pairs = [] # T
            elif self.vle == "nrtl-wohl":
                nrtl_pairs = [(0,1)]  # tau_12, tau_21 alpha
                if self.wohl_order == 3:
                    wohl_pairs = [(4,5)]
                elif self.wohl_order == 4:
                    wohl_pairs = [(4,5), (6,8)]
                elif self.wohl_order == 5:
                    wohl_pairs = [(4,5), (6,8), (9,12), (10,11)]
                else:
                    raise ValueError(f"Unsupported equivariant method with wohl order {self.wohl_order} for NRTL-Wohl.")
                self.output_equivariant_pairs = nrtl_pairs + wohl_pairs
                self.features_equivariant_pairs = []  # T
            elif self.vle == "activity":
                self.output_equivariant_pairs = [(0,1)]
                self.features_equivariant_pairs = [(0,1)] # x1, x2, T
            elif self.vle == "basic":
                self.output_equivariant_pairs = [(0,1)] # y1, y2, log10P
                self.features_equivariant_pairs = [(0,1), (3,4)] # x1, x2, T, log10p1sat, log10p2sat
            elif self.vle == "freestyle":
                self.output_equivariant_pairs = []
                self.features_equivariant_pairs = [(0,1)] # x1, x2, T
            else:
                raise ValueError(f"Unsupported equivariant method with vle {self.vle}.")

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)

        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:  # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:  # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        first_linear_dim = args.hidden_size * args.number_of_molecules
        if args.use_input_features:
            first_linear_dim += args.features_size
        if self.binary_equivariant:
            first_linear_dim = 2 * args.hidden_size + args.features_size # 2 molecules + features

        # Create FFN layers
        self.readout = build_ffn(
            first_linear_dim=first_linear_dim,
            hidden_size=args.ffn_hidden_size,
            num_layers=args.ffn_num_layers,
            output_size=self.output_size,
            dropout=args.dropout,
            activation=args.activation,
            dataset_type=args.dataset_type,
            spectra_activation=args.spectra_activation,
        )
        if self.vle in ["wohl", "nrtl-wohl"]:
            self.wohl_q = build_ffn(
                first_linear_dim=self.hidden_size + 1, # +1 for temperature only
                hidden_size=args.ffn_hidden_size,
                num_layers=args.ffn_num_layers,
                output_size=1, # q
                dropout=args.dropout,
                activation=args.activation,
                dataset_type=args.dataset_type,
                spectra_activation=args.spectra_activation,
            )
        if self.vle is not None and self.vp is not None:
            self.intrinsic_vp = build_ffn(
                first_linear_dim=self.hidden_size + 1, # +1 for temperature only
                hidden_size=args.ffn_hidden_size,
                num_layers=args.ffn_num_layers,
                output_size=self.vp_output_size,
                dropout=args.dropout,
                activation=args.activation,
                dataset_type=args.dataset_type,
                spectra_activation=args.spectra_activation,
            )
        if self.vle == "uniquac":
            self.uniquac_pure_ffn = build_ffn(
                first_linear_dim=self.hidden_size + 1,  # +1 for temperature
                hidden_size=args.ffn_hidden_size,
                num_layers=args.ffn_num_layers,
                output_size=2,  # r and q
                dropout=args.dropout,
                activation=args.activation,
                dataset_type=args.dataset_type,
                spectra_activation=args.spectra_activation,
            )
            if self.learn_uniquac_z:
                self.uniquac_z_ffn = build_ffn(
                    first_linear_dim=2*self.hidden_size,
                    hidden_size=args.ffn_hidden_size,
                    num_layers=args.ffn_num_layers,
                    output_size=1,  # Z
                    dropout=args.dropout,
                    activation=args.activation,
                    dataset_type=args.dataset_type,
                    spectra_activation=args.spectra_activation,
                )

    def fingerprint(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        fingerprint_type: str = "MPN",
    ) -> torch.Tensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == "MPN":
            return self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
        elif fingerprint_type == "last_FFN":
            return self.readout[:-1](
                self.encoder(
                    batch,
                    features_batch,
                    atom_descriptors_batch,
                    atom_features_batch,
                    bond_descriptors_batch,
                    bond_features_batch,
                )
            )
        else:
            raise ValueError(f"Unsupported fingerprint type {fingerprint_type}.")

    def forward(
        self, 
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        constraints_batch: List[torch.Tensor] = None,
        bond_types_batch: List[torch.Tensor] = None,
        hybrid_model_features_batch: List[np.ndarray] = None,
        get_parameters: bool = False,
    ) -> torch.Tensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param constraints_batch: A list of PyTorch tensors which applies constraint on atomic/bond properties.
        :param bond_types_batch: A list of PyTorch tensors storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions.
        """
        if hybrid_model_features_batch is not None:
            hybrid_model_features_batch = torch.from_numpy(np.array(hybrid_model_features_batch, dtype=np.float64)).float().to(self.device)

        features_batch = torch.from_numpy(np.array(features_batch, dtype=np.float64)).float().to(self.device)
        # get temperature and x for use in parameterized equations
        if self.vle in ["basic", "activity"]:
            output_temperature_batch = hybrid_model_features_batch[:,[2]]
            input_temperature_batch = features_batch[:,[2]]
            x_1 = hybrid_model_features_batch[:,[0]]
            x_2 = hybrid_model_features_batch[:,[1]]
        elif self.vle is not None:
            output_temperature_batch = hybrid_model_features_batch[:,[2]]
            input_temperature_batch = features_batch[:,[0]]
            x_1 = hybrid_model_features_batch[:,[0]]
            x_2 = hybrid_model_features_batch[:,[1]]
        elif self.vp is not None:
            output_temperature_batch = hybrid_model_features_batch[:,[0]]
            input_temperature_batch = features_batch[:,[0]]
            x_1 = None
            x_2 = None
        else:
            output_temperature_batch = None            
            input_temperature_batch = None
            x_1 = None
            x_2 = None

        # get Tc and Pc
        Tc = None
        log10Pc = None

        if self.noisy_temperature is not None and self.training:
            # noise is applied to the features temperature not the temperature batch
            noise_batch = np.random.randn(len(features_batch)) * self.noisy_temperature
            input_temperature_batch += noise_batch

        if self.vle == "freestyle":
            features_batch.requires_grad_(True)  # Enable gradient tracking for autograd

        # Make the encodings
        encodings = self.encoder(
            batch,
            features_batch,
            atom_descriptors_batch,
            atom_features_batch,
            bond_descriptors_batch,
            bond_features_batch,
        )

        if self.vle is not None:
            encoding_1 = encodings[:,:self.hidden_size] # first molecule
            encoding_2 = encodings[:,self.hidden_size:2*self.hidden_size] # second molecule

        # readout section
        if self.binary_equivariant:
            output = binary_equivariant_readout(encoding_1, encoding_2, features_batch, self.readout, self.output_equivariant_pairs, self.features_equivariant_pairs)
        else:
            output = self.readout(encodings)

        # self activity corrections
        if self.self_activity_correction or self.self_activity_lambda > 0:
            if self.vle == "basic":
                output_1, output_2 = None, None
            if self.binary_equivariant:
                output_1 = binary_equivariant_readout(encoding_1, encoding_1, features_batch, self.readout, self.output_equivariant_pairs, self.features_equivariant_pairs)
                output_2 = binary_equivariant_readout(encoding_2, encoding_2, features_batch, self.readout, self.output_equivariant_pairs, self.features_equivariant_pairs)
            else: # assumes there are no features that need to be swapped, only features T or [x1,x2,T]
                output_1 = self.readout(torch.cat([encoding_1, encoding_1, features_batch], axis=1))
                output_2 = self.readout(torch.cat([encoding_2, encoding_2, features_batch], axis=1))

        if self.vle in ["wohl", "nrtl-wohl"]:
            q_1 = nn.functional.softplus(self.wohl_q(torch.cat([encoding_1, input_temperature_batch], axis=1)))
            q_2 = nn.functional.softplus(self.wohl_q(torch.cat([encoding_2, input_temperature_batch], axis=1)))
        if self.vle == "uniquac":
            r1, q1 = torch.chunk(nn.functional.softplus(self.uniquac_pure_ffn(torch.cat([encoding_1, input_temperature_batch], dim=1))), 2, dim=1)
            r2, q2 = torch.chunk(nn.functional.softplus(self.uniquac_pure_ffn(torch.cat([encoding_2, input_temperature_batch], dim=1))), 2, dim=1)
       
        if self.vle is not None and self.vp is not None: # internal VP prediction
            vp1_output = self.intrinsic_vp(torch.cat([encoding_1, input_temperature_batch], axis=1))
            vp2_output = self.intrinsic_vp(torch.cat([encoding_2, input_temperature_batch], axis=1))
            log10p1sat = forward_vp(self.vp, vp1_output, output_temperature_batch)
            log10p2sat = forward_vp(self.vp, vp2_output, output_temperature_batch)
        elif self.vle is not None: # tabulated VP
            log10p1sat = hybrid_model_features_batch[:,[3]]
            log10p2sat = hybrid_model_features_batch[:,[4]]
        else:
            log10p1sat = None
            log10p2sat = None

        # get parameters if not full predicting
        if get_parameters:
            names, parameters = [], torch.empty(len(output), 0, device=self.device)
            if self.vle == "uniquac":
                if self.learn_uniquac_z:
                    Z = torch.floor(nn.functional.softplus(self.uniquac_z_ffn(encodings))) + 8  # Ensure Z is a positive integer >= 8
                else:
                    Z = torch.full((len(output), 1), self.uniquac_z, device=self.device)
                tau12, tau21 = torch.chunk(nn.functional.softplus(output), 2, dim=1)  # Apply softplus to ensure positive tau values
                r1, q1 = torch.chunk(nn.functional.softplus(self.uniquac_pure_ffn(torch.cat([encoding_1, input_temperature_batch], dim=1))), 2, dim=1)
                r2, q2 = torch.chunk(nn.functional.softplus(self.uniquac_pure_ffn(torch.cat([encoding_2, input_temperature_batch], dim=1))), 2, dim=1)
                act_names, act_parameters = get_uniquac_parameters(output, r1, r2, q1, q2, x_1, x_2, input_temperature_batch, Z)
                names += act_names
                parameters = torch.cat([parameters, act_parameters], dim=1)
                
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    act1_names, act1_parameters = get_uniquac_parameters(output, r1, r2, q1, q2, x_1, x_1, input_temperature_batch, Z, 1)
                    act2_names, act2_parameters = get_uniquac_parameters(output, r1, r2, q1, q2, x_2, x_2, input_temperature_batch, Z, 2)
                    names += act1_names + act2_names
                    parameters = torch.cat([parameters, act1_parameters, act2_parameters], dim=1)

            elif self.vle == "wohl":
                act_names, act_parameters = get_wohl_parameters(output, self.wohl_order, q_1, q_2)
                names += act_names
                parameters = torch.cat([parameters, act_parameters], axis=1)
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    act1_names, act1_parameters = get_wohl_parameters(output_1, self.wohl_order, q_1, q_1, 1)
                    act2_names, act2_parameters = get_wohl_parameters(output_2, self.wohl_order, q_2, q_2, 2)
                    names += act1_names + act2_names
                    parameters = torch.cat([parameters, act1_parameters, act2_parameters], axis=1)
            elif self.vle == "nrtl-wohl":
                act_names, act_parameters = get_nrtl_wohl_parameters(output, self.wohl_order, q_1, q_2)
                names += act_names
                parameters = torch.cat([parameters, act_parameters], axis=1)
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    act1_names, act1_parameters = get_nrtl_wohl_parameters(output_1, self.wohl_order, q_1, q_1, 1)
                    act2_names, act2_parameters = get_nrtl_wohl_parameters(output_2, self.wohl_order, q_2, q_2, 2)
                    names += act1_names + act2_names
                    parameters = torch.cat([parameters, act1_parameters, act2_parameters], axis=1)
            elif self.vle == "nrtl":
                nrtl_names, nrtl_parameters = get_nrtl_parameters(output)
                names += nrtl_names
                parameters = torch.cat([parameters, nrtl_parameters], axis=1)
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    nrtl1_names, nrtl1_parameters = get_nrtl_parameters(output_1, 1)
                    nrtl2_names, nrtl2_parameters = get_nrtl_parameters(output_2, 2)
                    names += nrtl1_names + nrtl2_names
                    parameters = torch.cat([parameters, nrtl1_parameters, nrtl2_parameters], axis=1)
            elif self.vle == "activity":
                act_names, act_parameters = get_activity_parameters(output)
                names += act_names
                parameters = torch.cat([parameters, act_parameters], axis=1)
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    act1_names, act1_parameters = get_activity_parameters(output_1, 1)
                    act2_names, act2_parameters = get_activity_parameters(output_2, 2)
                    names += act1_names + act2_names
                    parameters = torch.cat([parameters, act1_parameters, act2_parameters], axis=1)

            if self.vle is not None and self.vp is not None and self.vp != "basic":
                vp1_names = get_vp_parameter_names(self.vp, 1)
                vp2_names = get_vp_parameter_names(self.vp, 2)
                names += vp1_names + vp2_names
                parameters = torch.cat([parameters, vp1_output, vp2_output], axis=1)
            elif self.vp is not None and self.vp != "basic":
                vp_names = get_vp_parameter_names(self.vp)
                names += vp_names
                parameters = torch.cat([parameters, output], axis=1)
            return names, parameters

        # VLE models
        if self.vle == "basic":
            output = forward_vle_basic(output)
            regularization = 0
        elif self.vle is not None:
            if self.vle == "activity":
                ln_gamma_1, ln_gamma_2 = forward_vle_activity(output=output)
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    ln_gamma_1_1, ln_gamma_2_1 = forward_vle_activity(output=output_1)
                    ln_gamma_1_2, ln_gamma_2_2 = forward_vle_activity(output=output_2)
            elif self.vle == "wohl":
                ln_gamma_1, ln_gamma_2 = forward_vle_wohl(output=output, wohl_order=self.wohl_order, x_1=x_1, x_2=x_2, q_1=q_1, q_2=q_2)
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    ln_gamma_1_1, ln_gamma_2_1 = forward_vle_wohl(output=output_1, wohl_order=self.wohl_order, x_1=x_1, x_2=x_2, q_1=q_1, q_2=q_1)
                    ln_gamma_1_2, ln_gamma_2_2 = forward_vle_wohl(output=output_2, wohl_order=self.wohl_order, x_1=x_1, x_2=x_2, q_1=q_2, q_2=q_2)
            elif self.vle == "nrtl":
                ln_gamma_1, ln_gamma_2 = forward_vle_nrtl(output=output, x_1=x_1, x_2=x_2)
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    ln_gamma_1_1, ln_gamma_2_1 = forward_vle_nrtl(output=output_1, x_1=x_1, x_2=x_2)
                    ln_gamma_1_2, ln_gamma_2_2 = forward_vle_nrtl(output=output_2, x_1=x_1, x_2=x_2)
            elif self.vle == "nrtl-wohl":
                ln_gamma_1, ln_gamma_2 = forward_vle_nrtl_wohl(output=output, x_1=x_1, x_2=x_2, q_1=q_1, q_2=q_2, wohl_order=self.wohl_order)
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    ln_gamma_1_1, ln_gamma_2_1 = forward_vle_nrtl_wohl(output=output_1, x_1=x_1, x_2=x_2, q_1=q_1, q_2=q_1, wohl_order=self.wohl_order)
                    ln_gamma_1_2, ln_gamma_2_2 = forward_vle_nrtl_wohl(output=output_2, x_1=x_1, x_2=x_2, q_1=q_2, q_2=q_2, wohl_order=self.wohl_order)
            elif self.vle == "uniquac":
                if self.learn_uniquac_z:
                    Z = torch.floor(nn.functional.softplus(self.uniquac_z_ffn(encodings))) + 8  # Ensure Z is a positive integer >= 8
                else:
                    Z = torch.full((len(output), 1), self.uniquac_z, device=self.device)
                
                tau12, tau21 = torch.chunk(nn.functional.softplus(output), 2, dim=1)  # Apply softplus to ensure positive tau values
                r1, q1 = torch.chunk(nn.functional.softplus(self.uniquac_pure_ffn(torch.cat([encoding_1, input_temperature_batch], dim=1))), 2, dim=1)
                r2, q2 = torch.chunk(nn.functional.softplus(self.uniquac_pure_ffn(torch.cat([encoding_2, input_temperature_batch], dim=1))), 2, dim=1)
                uniquac_params = torch.cat([tau12, tau21, r1, r2, q1, q2], dim=1)
                ln_gamma_1, ln_gamma_2 = forward_vle_uniquac(uniquac_params, x_1, x_2, input_temperature_batch, Z)
                
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    ln_gamma_1_1, ln_gamma_2_1 = forward_vle_uniquac(uniquac_params, x_1, x_1, input_temperature_batch, Z)
                    ln_gamma_1_2, ln_gamma_2_2 = forward_vle_uniquac(uniquac_params, x_2, x_2, input_temperature_batch, Z)
            elif self.vle == "freestyle":
                ln_gamma_1, ln_gamma_2 = forward_vle_freestyle(output=output, features=features_batch)
                if self.self_activity_correction or self.self_activity_lambda > 0:
                    ln_gamma_1_1, ln_gamma_2_1 = forward_vle_freestyle(output=output_1, features=features_batch)
                    ln_gamma_1_2, ln_gamma_2_2 = forward_vle_freestyle(output=output_2, features=features_batch)
            else:
                raise ValueError(f"Unsupported VLE model {self.vle}.")
            
            if self.self_activity_correction:
                # M1(A,B)* = M1(A,B) - x1 * M1(A,A) - x2 * M1(B,B)
                ln_gamma_1 = ln_gamma_1 - x_1 * ln_gamma_1_1 - x_2 * ln_gamma_1_2
                # M2(A,B)* = M2(A,B) - x1 * M2(A,A) - x2 * M2(B,B)
                ln_gamma_2 = ln_gamma_2 - x_1 * ln_gamma_2_1 - x_2 * ln_gamma_2_2
            if self.self_activity_lambda > 0:
                regularization = self.self_activity_lambda * (torch.sum(ln_gamma_1_1**2) + torch.sum(ln_gamma_1_2**2) + torch.sum(ln_gamma_2_1**2) + torch.sum(ln_gamma_2_2**2))

            # create output
            lnp1sat = log10p1sat * np.log(10)
            lnp2sat = log10p2sat * np.log(10)
            ln_P1 = lnp1sat + torch.log(x_1) + ln_gamma_1
            ln_P2 = lnp2sat + torch.log(x_2) + ln_gamma_2
            lnP = torch.logaddexp(ln_P1, ln_P2)
            y_1 = torch.exp(ln_P1 - lnP)
            y_2 = torch.exp(ln_P2 - lnP)
            log10P = lnP / np.log(10)

            output = torch.cat([y_1, y_2, log10P, ln_gamma_1, ln_gamma_2, log10p1sat, log10p2sat], axis=1)
        # VP
        if self.vp is not None and self.vle is None:
            output = forward_vp(self.vp, output, output_temperature_batch, Tc, log10Pc)
        
        if self.self_activity_lambda > 0 and self.training:
            output = (output, regularization)
        return output
