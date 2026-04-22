from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class AEConfig:
    # Guiding text prompt
    prompt: str
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = None
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42])
    # Path to save all outputs to
    output_path: Path = Path('./outputs')
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 25
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)


@dataclass
class CONFORMConfig:
    num_inference_steps = 50 # Number of steps to run the model
    guidance_scale = 7.5 # Guidance scale for diffusion
    attn_res = (32, 32) # Resolution of the attention map to apply CONFORM
    steps_to_save_attention_maps = list(range(num_inference_steps)) # Steps to save attention maps
    max_iter_to_alter = 25 # Which steps to stop updating the latents
    refinement_steps = 20 # Number of refinement steps
    scale_factor = 20 # Scale factor for the optimization step
    iterative_refinement_steps = [0, 10, 20] # Iterative refinement steps
    do_smoothing = True # Apply smoothing to the attention maps
    smoothing_sigma = 0.5 # Sigma for the smoothing kernel
    smoothing_kernel_size = 3 # Kernel size for the smoothing kernel
    temperature = 0.5 # Temperature for the contrastive loss
    softmax_normalize = True # Normalize the attention maps
    softmax_normalize_attention_maps = False # Normalize the attention maps
    add_previous_attention_maps = True # Add previous attention maps to the loss calculation
    previous_attention_map_anchor_step = None # Use a specific step as the previous attention map
    loss_fn = "ntxent" # Loss function to use
