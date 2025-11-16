# Neural Style Transfer via Diffusion Latent Steering

This document describes the implementation of neural style transfer for protein structure prediction using diffusion latent steering in OpenFold 3.

## Overview

Neural style transfer is a technique that optimizes internal model representations to satisfy multiple objectives simultaneously. In the context of protein structure prediction, we optimize the **pairwise latent representation (zij)** to balance:

1. **Content Objective**: Match predicted structure to a reference structure (minimize RMSD)
2. **Style Objective**: Maximize confidence scores (e.g., iPTM, pTM, pLDDT)

This approach enables fine-grained control over the structure generation process by manipulating latents before they reach the diffusion head.

## Architecture Modifications

### 1. Model Extensions (`openfold3/projects/of3_all_atom/model.py`)

Two new methods were added to the `OpenFold3` class:

#### `export_latents(batch, num_recycles=None)`

Exports internal latent representations before the diffusion head.

**Arguments:**
- `batch`: Input feature dictionary
- `num_recycles`: Number of recycles (default: from config)

**Returns:**
- `si_input`: [N_token, C_s_input] Single input representation
- `si_trunk`: [N_token, C_s] Single representation from trunk
- `zij_trunk`: [N_token, N_token, C_z] **Pairwise representation (optimization target)**

**Example:**
```python
si_input, si_trunk, zij_trunk = model.export_latents(batch)
print(f"Pairwise latent shape: {zij_trunk.shape}")  # [*, 384, 384, 128]
```

#### `run_from_latents(batch, si_input, si_trunk, zij_trunk, no_rollout_steps=None, no_rollout_samples=1)`

Runs diffusion and confidence modules from provided latents.

**Arguments:**
- `batch`: Input feature dictionary
- `si_input`, `si_trunk`, `zij_trunk`: Latent representations
- `no_rollout_steps`: Number of diffusion steps (default: full rollout)
- `no_rollout_samples`: Number of samples to generate

**Returns:**
- Output dictionary containing:
  - `atom_positions_predicted`: [N_samples, N_atom, 3]
  - `pae_logits`: [N_token, N_token, 64]
  - `plddt_logits`: [N_atom, 50]
  - Other confidence head outputs

**Example:**
```python
# Modify zij_trunk as desired
zij_modified = zij_trunk * 1.1  # Example modification

# Generate predictions from modified latents
output = model.run_from_latents(
    batch=batch,
    si_input=si_input,
    si_trunk=si_trunk,
    zij_trunk=zij_modified,
    no_rollout_samples=5
)
```

## Neural Style Transfer Implementation

### Core Components

The `NeuralStyleTransfer` class (`scripts/neural_style_transfer.py`) implements gradient-based optimization:

```python
from scripts.neural_style_transfer import NeuralStyleTransfer

style_transfer = NeuralStyleTransfer(
    model=model,
    batch=batch,
    reference_pdb_path='path/to/reference.pdb',
    content_weight=1.0,
    style_weight=0.1,
    learning_rate=0.01,
    num_iterations=100,
    content_atom_selection='CA',  # 'CA', 'all', or 'backbone'
    style_metric='iptm',  # 'iptm' or 'ptm'
    device='cuda'
)

optimized_zij, history = style_transfer.optimize()
style_transfer.save_results(optimized_zij, './output')
```

### Loss Functions

#### Content Loss: Coordinate RMSD

Minimizes the root-mean-square deviation between predicted and reference structures:

```python
def compute_content_loss(predicted_coords, reference_coords):
    # Center structures
    pred_centered = predicted_coords - predicted_coords.mean(dim=0)
    ref_centered = reference_coords - reference_coords.mean(dim=0)

    # Compute RMSD
    rmsd = torch.sqrt(torch.mean((pred_centered - ref_centered) ** 2))
    return rmsd
```

**Atom Selection Options:**
- `CA`: Only C-alpha atoms (fast, standard for fold comparison)
- `backbone`: N, CA, C, O atoms (more detailed)
- `all`: All atoms (most detailed, slowest)

#### Style Loss: Confidence Score

Maximizes confidence metrics from the model's auxiliary heads:

```python
def compute_style_loss(output, metric='iptm'):
    from openfold3.core.metrics.confidence import compute_ptm

    pae_logits = output['pae_logits']
    token_mask = batch['token_mask']

    ptm_score = compute_ptm(
        logits=pae_logits,
        has_frame=token_mask.unsqueeze(0),
        bin_min=0,
        bin_max=32,
        no_bins=64,
        mask_i=token_mask.bool(),
        asym_id=batch.get('asym_id'),
        interface=(metric == 'iptm'),
    )

    # Return negative to maximize (minimize negative)
    return -ptm_score
```

**Confidence Metric Options:**
- `iptm`: Interface Predicted TM-score (for multi-chain complexes)
- `ptm`: Predicted TM-score (overall structure quality)

### Optimization Loop

The optimization proceeds as follows:

```python
# Initialize optimizable parameter
zij_opt = zij_trunk.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([zij_opt], lr=0.01)

for iteration in range(num_iterations):
    optimizer.zero_grad()

    # Forward pass with current zij
    output = model.run_from_latents(
        batch=batch,
        si_input=si_input,
        si_trunk=si_trunk,
        zij_trunk=zij_opt,
        no_rollout_samples=1
    )

    # Compute losses
    loss_content = compute_content_loss(output)
    loss_style = compute_style_loss(output)

    # Combined objective
    total_loss = content_weight * loss_content + style_weight * loss_style

    # Optimize
    total_loss.backward()
    optimizer.step()
```

## Usage Guide

### Step 1: Prepare Inputs

You need:
1. A trained OpenFold3 model
2. Input features for your target sequence
3. A reference PDB file for the content objective

```python
# Load model
from openfold3.projects.of3_all_atom.model import OpenFold3
from openfold3.projects.of3_all_atom.config.model_config import get_config

config = get_config()
model = OpenFold3(config)
checkpoint = torch.load("model.pt")
model.load_state_dict(checkpoint)
model.eval()

# Prepare batch (implementation-specific)
batch = prepare_batch_from_sequence(sequence="MKTAYIAKQR...")

# Specify reference PDB
reference_pdb = "path/to/reference_structure.pdb"
```

### Step 2: Run Style Transfer

```python
from scripts.neural_style_transfer import NeuralStyleTransfer

style_transfer = NeuralStyleTransfer(
    model=model,
    batch=batch,
    reference_pdb_path=reference_pdb,
    content_weight=1.0,      # Balance content vs style
    style_weight=0.1,        # Lower weight = less confidence optimization
    learning_rate=0.01,      # Adam learning rate
    num_iterations=100,      # Number of gradient steps
    content_atom_selection='CA',
    style_metric='iptm',
    device='cuda'
)

# Optimize
optimized_zij, history = style_transfer.optimize()
```

### Step 3: Analyze Results

```python
import matplotlib.pyplot as plt

# Plot loss curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.plot(history['total_loss'])
plt.title('Total Loss')
plt.xlabel('Iteration')

plt.subplot(1, 3, 2)
plt.plot(history['content_loss'])
plt.title('Content Loss (RMSD)')
plt.xlabel('Iteration')

plt.subplot(1, 3, 3)
plt.plot([-x for x in history['style_loss']])  # Negate for actual score
plt.title('Style Score (iPTM)')
plt.xlabel('Iteration')

plt.tight_layout()
plt.savefig('optimization_history.png')
```

### Step 4: Generate Final Structures

```python
# Save results
style_transfer.save_results(optimized_zij, './output')

# Load saved results
results = torch.load('./output/final_prediction.pt')
final_coords = results['atom_positions_predicted']  # [1, N_samples, N_atom, 3]

# Multiple samples from optimized latents
print(f"Generated {final_coords.shape[1]} structural samples")
```

## Advanced Usage

### Custom Objective Functions

You can extend the `NeuralStyleTransfer` class with custom objectives:

```python
class CustomStyleTransfer(NeuralStyleTransfer):
    def compute_style_loss(self, output):
        # Custom objective: maximize pLDDT instead of iPTM
        plddt_logits = output['plddt_logits']
        plddt = torch.softmax(plddt_logits, dim=-1)

        # Compute expected pLDDT
        bin_centers = torch.linspace(0, 1, 50, device=plddt.device)
        expected_plddt = torch.sum(plddt * bin_centers, dim=-1)

        # Return negative to maximize
        return -expected_plddt.mean()
```

### Multi-Objective Optimization

Combine multiple objectives:

```python
class MultiObjectiveStyleTransfer(NeuralStyleTransfer):
    def __init__(self, *args, plddt_weight=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.plddt_weight = plddt_weight

    def optimize(self):
        # ... setup ...

        for iteration in range(self.num_iterations):
            output = self.model.run_from_latents(...)

            # Multiple objectives
            loss_content = self.compute_content_loss(output)
            loss_iptm = self.compute_style_loss(output)  # iPTM
            loss_plddt = self.compute_plddt_loss(output)

            total_loss = (
                self.content_weight * loss_content +
                self.style_weight * loss_iptm +
                self.plddt_weight * loss_plddt
            )

            # ... optimize ...
```

### Optimizing Other Latents

While we focus on `zij_trunk` (pairwise latents), you can also optimize:

- `si_trunk`: Single representation (token-level features)
- Both `si_trunk` and `zij_trunk` simultaneously

```python
# Optimize both single and pair representations
si_opt = si_trunk.clone().detach().requires_grad_(True)
zij_opt = zij_trunk.clone().detach().requires_grad_(True)

optimizer = torch.optim.Adam([
    {'params': [si_opt], 'lr': 0.005},
    {'params': [zij_opt], 'lr': 0.01}
])
```

## Technical Details

### Gradient Flow

The gradient flows through:
1. **Pairwise latent (zij)** → optimization target
2. **Diffusion conditioning** → conditions diffusion on zij
3. **Diffusion transformer** → processes conditioned latents
4. **Atom attention decoder** → generates atom positions
5. **Auxiliary heads** → compute confidence scores
6. **Loss functions** → provide supervision signal

### Memory Considerations

- **Gradient checkpointing**: Disabled during optimization for speed
- **Sample size**: Use `no_rollout_samples=1` during optimization, increase for final generation
- **Batch size**: Process one structure at a time to reduce memory
- **Device offloading**: Automatic for large structures

### Hyperparameter Tuning

**Learning Rate:**
- Start with `0.01`
- Decrease if optimization is unstable
- Increase if convergence is too slow

**Content Weight:**
- `1.0` is a good default
- Increase to prioritize matching reference structure
- Decrease to allow more deviation

**Style Weight:**
- `0.1` is a good default
- Too high may ignore content constraint
- Too low may not improve confidence

**Iterations:**
- `100` iterations typical for convergence
- Monitor loss curves to determine early stopping

## Limitations and Future Work

### Current Limitations

1. **Atom selection**: CA-only filtering is simplified; proper atom name mapping needed
2. **Model loading**: Checkpoint loading not fully implemented in CLI
3. **Batch preparation**: Input feature creation requires integration with data pipeline
4. **Alignment**: No automatic alignment between predicted and reference structures

### Future Enhancements

1. **Procrustes alignment**: Add automatic superposition for RMSD computation
2. **Selective optimization**: Optimize only specific regions of zij
3. **Regularization**: Add L2 penalty to prevent large latent deviations
4. **Multi-reference**: Blend multiple reference structures
5. **Online visualization**: Real-time structure visualization during optimization

## References

- AlphaFold 3 Paper: Abramson et al. (2024)
- AlphaFold 3 Supplement: Section 5.9 (Confidence Metrics)
- OpenFold 3 Documentation: https://github.com/aqlaboratory/openfold

## Citation

If you use this neural style transfer implementation, please cite:

```bibtex
@software{openfold3_style_transfer,
  title = {Neural Style Transfer via Diffusion Latent Steering for OpenFold 3},
  author = {AlQuraishi Laboratory},
  year = {2025},
  url = {https://github.com/aqlaboratory/openfold}
}
```
