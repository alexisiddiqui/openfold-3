#!/usr/bin/env python
# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Neural style transfer via diffusion latent steering for protein structure prediction.

This script performs gradient-based optimization on the pairwise latent representation (z)
to satisfy dual objectives:
  - Content loss: Minimize CA coordinate RMSD to a reference PDB structure
  - Style loss: Maximize a confidence score (e.g., iPTM)

The optimization manipulates internal latents before the diffusion head, enabling
fine-grained control over the structure generation process.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Literal

import biotite.structure as bs
import biotite.structure.io.pdb as pdb
import numpy as np
import torch
import torch.nn.functional as F
from ml_collections import ConfigDict

from openfold3.core.metrics.confidence import compute_ptm
from openfold3.projects.of3_all_atom.model import OpenFold3
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtomRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralStyleTransfer:
    """
    Neural style transfer via diffusion latent steering.

    Optimizes the pairwise latent representation (z) to balance:
    - Content objective: Match predicted structure to reference coordinates
    - Style objective: Maximize confidence scores (e.g., iPTM, pTM)
    """

    def __init__(
        self,
        model: OpenFold3,
        batch: dict,
        reference_pdb_path: str,
        content_weight: float = 1.0,
        style_weight: float = 0.1,
        learning_rate: float = 0.01,
        num_iterations: int = 100,
        content_atom_selection: Literal["CA", "all", "backbone"] = "CA",
        style_metric: Literal["iptm", "ptm"] = "iptm",
        device: str = "cuda",
    ):
        """
        Args:
            model: Pretrained OpenFold3 model
            batch: Input feature dictionary
            reference_pdb_path: Path to reference PDB file for content loss
            content_weight: Weight for content loss term
            style_weight: Weight for style loss term
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of optimization iterations
            content_atom_selection: Which atoms to use for content loss ("CA", "all", "backbone")
            style_metric: Which confidence metric to maximize ("iptm", "ptm")
            device: Device to run on ("cuda" or "cpu")
        """
        self.model = model
        self.model.eval()
        self.batch = batch
        self.reference_pdb_path = reference_pdb_path
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.content_atom_selection = content_atom_selection
        self.style_metric = style_metric
        self.device = device

        # Move batch to device
        self._move_batch_to_device()

        # Load reference structure
        self.reference_coords = self._load_reference_pdb()

        # Export initial latents (these will be optimized)
        logger.info("Exporting initial latents from model trunk...")
        with torch.no_grad():
            self.si_input, self.si_trunk, self.zij_trunk = model.export_latents(batch)

        logger.info(f"Latent shapes - si_input: {self.si_input.shape}, "
                   f"si_trunk: {self.si_trunk.shape}, zij_trunk: {self.zij_trunk.shape}")

    def _move_batch_to_device(self):
        """Move all tensors in batch to the specified device."""
        def move_to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            return x

        def recursive_apply(d):
            if isinstance(d, dict):
                return {k: recursive_apply(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple)):
                return type(d)(recursive_apply(item) for item in d)
            else:
                return move_to_device(d)

        self.batch = recursive_apply(self.batch)

    def _load_reference_pdb(self) -> torch.Tensor:
        """
        Load reference PDB and extract coordinates for selected atoms.

        Returns:
            reference_coords: [N_atom, 3] tensor of reference coordinates
        """
        logger.info(f"Loading reference PDB from {self.reference_pdb_path}")

        with open(self.reference_pdb_path, "r") as f:
            pdb_file = pdb.PDBFile.read(f)

        structure = pdb_file.get_structure(model=1)

        # Filter to selected atoms
        if self.content_atom_selection == "CA":
            atom_mask = structure.atom_name == "CA"
        elif self.content_atom_selection == "backbone":
            atom_mask = np.isin(structure.atom_name, ["N", "CA", "C", "O"])
        else:  # "all"
            atom_mask = np.ones(len(structure), dtype=bool)

        filtered_structure = structure[atom_mask]
        coords = filtered_structure.coord

        logger.info(f"Loaded {len(coords)} {self.content_atom_selection} atoms from reference PDB")

        return torch.tensor(coords, dtype=torch.float32, device=self.device)

    def _get_predicted_coords(self, output: dict) -> torch.Tensor:
        """
        Extract predicted coordinates for selected atoms from model output.

        Args:
            output: Model output dictionary

        Returns:
            predicted_coords: [N_atom, 3] tensor of predicted coordinates
        """
        # Get predicted atom positions [1, N_samples, N_atom, 3]
        atom_positions = output["atom_positions_predicted"]

        # Take first sample
        coords = atom_positions[0, 0]  # [N_atom, 3]

        # Filter to selected atoms based on atom_mask
        atom_mask = self.batch["atom_mask"][0]  # [N_atom]

        if self.content_atom_selection == "CA":
            # Extract CA atoms based on atom names or indices
            # For simplicity, assume every 3rd atom is CA in backbone (N, CA, C pattern)
            # This is a simplification - in practice you'd need proper atom name mapping
            # TODO: Implement proper atom name-based filtering
            ca_indices = torch.arange(1, coords.shape[0], 3, device=self.device)
            coords = coords[ca_indices]
        elif self.content_atom_selection == "backbone":
            # Extract N, CA, C, O atoms
            # TODO: Implement proper filtering
            pass

        return coords

    def compute_content_loss(self, output: dict) -> torch.Tensor:
        """
        Compute content loss as RMSD between predicted and reference coordinates.

        Args:
            output: Model output dictionary

        Returns:
            content_loss: Scalar tensor representing coordinate RMSD
        """
        predicted_coords = self._get_predicted_coords(output)

        # Ensure shapes match
        min_len = min(len(predicted_coords), len(self.reference_coords))
        predicted_coords = predicted_coords[:min_len]
        reference_coords = self.reference_coords[:min_len]

        # Center both structures
        predicted_centered = predicted_coords - predicted_coords.mean(dim=0)
        reference_centered = reference_coords - reference_coords.mean(dim=0)

        # Compute RMSD
        rmsd = torch.sqrt(torch.mean((predicted_centered - reference_centered) ** 2))

        return rmsd

    def compute_style_loss(self, output: dict) -> torch.Tensor:
        """
        Compute style loss as negative confidence score (to maximize confidence).

        Args:
            output: Model output dictionary

        Returns:
            style_loss: Scalar tensor representing negative confidence score
        """
        # Get PAE logits for PTM/iPTM calculation
        pae_logits = output.get("pae_logits")

        if pae_logits is None:
            logger.warning("PAE logits not found in output, returning zero style loss")
            return torch.tensor(0.0, device=self.device)

        # Prepare inputs for compute_ptm
        # pae_logits shape: [1, N_token, N_token, 64]
        # has_frame: mask for tokens with valid frames
        token_mask = self.batch["token_mask"]  # [N_token]
        has_frame = token_mask.unsqueeze(0)  # [1, N_token]

        # Compute PTM or iPTM
        interface = self.style_metric == "iptm"
        asym_id = self.batch.get("asym_id") if interface else None

        ptm_score = compute_ptm(
            logits=pae_logits,
            has_frame=has_frame,
            bin_min=0,
            bin_max=32,
            no_bins=64,
            mask_i=token_mask.bool(),
            asym_id=asym_id,
            interface=interface,
        )

        # Return negative score to maximize (minimize negative)
        return -ptm_score

    def optimize(self):
        """
        Run gradient-based optimization on the pairwise latent (z).

        Returns:
            optimized_zij: Optimized pairwise latent tensor
            history: Dictionary containing loss history
        """
        # Make zij_trunk a parameter to optimize
        zij_trunk_opt = self.zij_trunk.clone().detach().requires_grad_(True)

        # Set up optimizer
        optimizer = torch.optim.Adam([zij_trunk_opt], lr=self.learning_rate)

        # History tracking
        history = {
            "total_loss": [],
            "content_loss": [],
            "style_loss": [],
        }

        logger.info(f"Starting optimization for {self.num_iterations} iterations...")
        logger.info(f"Content weight: {self.content_weight}, Style weight: {self.style_weight}")

        for iteration in range(self.num_iterations):
            optimizer.zero_grad()

            # Run diffusion and confidence modules with current zij
            output = self.model.run_from_latents(
                batch=self.batch,
                si_input=self.si_input,
                si_trunk=self.si_trunk,
                zij_trunk=zij_trunk_opt,
                no_rollout_samples=1,
            )

            # Compute losses
            content_loss = self.compute_content_loss(output)
            style_loss = self.compute_style_loss(output)

            # Combined loss
            total_loss = (
                self.content_weight * content_loss +
                self.style_weight * style_loss
            )

            # Backward pass
            total_loss.backward()

            # Gradient descent step
            optimizer.step()

            # Track history
            history["total_loss"].append(total_loss.item())
            history["content_loss"].append(content_loss.item())
            history["style_loss"].append(style_loss.item())

            # Logging
            if iteration % 10 == 0 or iteration == self.num_iterations - 1:
                logger.info(
                    f"Iteration {iteration}/{self.num_iterations} - "
                    f"Total Loss: {total_loss.item():.4f}, "
                    f"Content Loss: {content_loss.item():.4f}, "
                    f"Style Loss: {style_loss.item():.4f}"
                )

        logger.info("Optimization complete!")

        return zij_trunk_opt.detach(), history

    def save_results(self, optimized_zij: torch.Tensor, output_dir: str):
        """
        Save optimized latents and final structure prediction.

        Args:
            optimized_zij: Optimized pairwise latent tensor
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save optimized latents
        torch.save(
            {
                "si_input": self.si_input.cpu(),
                "si_trunk": self.si_trunk.cpu(),
                "zij_trunk": optimized_zij.cpu(),
            },
            output_dir / "optimized_latents.pt",
        )
        logger.info(f"Saved optimized latents to {output_dir / 'optimized_latents.pt'}")

        # Generate final prediction with optimized latents
        with torch.no_grad():
            final_output = self.model.run_from_latents(
                batch=self.batch,
                si_input=self.si_input,
                si_trunk=self.si_trunk,
                zij_trunk=optimized_zij,
                no_rollout_samples=5,  # Generate multiple samples
            )

        # Save final coordinates
        torch.save(
            {
                "atom_positions_predicted": final_output["atom_positions_predicted"].cpu(),
                "pae_logits": final_output.get("pae_logits", None),
                "plddt_logits": final_output.get("plddt_logits", None),
            },
            output_dir / "final_prediction.pt",
        )
        logger.info(f"Saved final prediction to {output_dir / 'final_prediction.pt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Neural style transfer via diffusion latent steering"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON file with protein features",
    )
    parser.add_argument(
        "--reference_pdb",
        type=str,
        required=True,
        help="Path to reference PDB file for content loss",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./style_transfer_output",
        help="Directory to save results",
    )
    parser.add_argument(
        "--content_weight",
        type=float,
        default=1.0,
        help="Weight for content loss (RMSD to reference)",
    )
    parser.add_argument(
        "--style_weight",
        type=float,
        default=0.1,
        help="Weight for style loss (confidence score)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--content_atom_selection",
        type=str,
        default="CA",
        choices=["CA", "all", "backbone"],
        help="Which atoms to use for content loss",
    )
    parser.add_argument(
        "--style_metric",
        type=str,
        default="iptm",
        choices=["iptm", "ptm"],
        help="Which confidence metric to maximize",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    # TODO: Implement proper model loading from checkpoint
    # This would typically involve:
    # 1. Loading config
    # 2. Initializing model
    # 3. Loading weights
    # For now, this is a placeholder
    raise NotImplementedError(
        "Model loading not implemented. Please implement model checkpoint loading."
    )

    # Load input features
    logger.info(f"Loading input features from {args.input_json}")
    # TODO: Implement input feature loading
    # This would parse the input JSON and create the batch dictionary
    raise NotImplementedError(
        "Input feature loading not implemented. Please implement batch creation from input JSON."
    )

    # Initialize style transfer
    style_transfer = NeuralStyleTransfer(
        model=model,
        batch=batch,
        reference_pdb_path=args.reference_pdb,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        content_atom_selection=args.content_atom_selection,
        style_metric=args.style_metric,
        device=args.device,
    )

    # Run optimization
    optimized_zij, history = style_transfer.optimize()

    # Save results
    style_transfer.save_results(optimized_zij, args.output_dir)

    logger.info("Neural style transfer complete!")


if __name__ == "__main__":
    main()
