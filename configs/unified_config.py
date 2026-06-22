from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class UnifiedConfig:
    """Unified configuration for Flow Matching, DDPM, VAE, and GAN."""

    # Method control
    method: str = "flowmatching"  # flowmatching / ddpm / vae / gan / gan
    mode: str = "train_sample"    # train_sample / sample_only
    resume_ckpt: str = ""
    skip_eval: bool = False

    # Data paths
    ehr_csv: str = "your_ehr_csv_datapath"
    sc_csv: str = "your_single_cell_csv_datapath"
    label_xlsx: str = "your_label_xlsx_datapath"
    clip_cfg: str = "./checkpoints/clip_config.json"
    clip_ckpt: str = "./checkpoints/bestCLIP_model.pth"

    # Runtime
    batch_size: int = 1
    lr: float = 1e-4
    lr_d: float = 1e-4
    epochs: int = 1000
    early_stop: int = 20
    seed: int = 42
    num_workers: int = 0
    compile_model: bool = True

    # Split
    train_val_limit: int = 1400
    train_ratio: float = 0.7

    # Shared backbone
    feature_dims: int = 36
    condition_mode: str = "cls"
    condition_token_keep_all_max: float | None = None
    condition_token_dropout_enabled: bool = False
    model_dims: int = 128
    dims_mult: Tuple[int, ...] = (1, 2, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (2, 4, 8, 16)
    dropout: float = 0.0
    dropout_attn: float = 0.1
    num_heads: int = 4

    # Scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    min_lr: float = 1e-6

    # Sampling target control
    sample_source: str = "train"
    sample_donor_ids: Tuple[str, ...] = field(default_factory=tuple)
    num_cells_total: int = 1000
    cell_num_per_sample: int = 1000

    # Flow Matching
    flow_method: str = "euler"
    flow_euler_steps: int = 100
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-5
    ode_solver: str = "dopri5"

    # DDPM
    ddpm_pred_type: str = "x0"
    ddpm_sampler: str = "ddpm"
    ddim_steps: int = 50
    eta: float = 0.0
    clip_denoised: bool = True
    diffusion_steps: int = 1000
    beta_schedule: str = "linear"

    # VAE
    latent_dim: int = 128
    kl_weight: float = 0.1

    # GAN
    n_critic: int = 5
    lambda_gp: float = 10.0
    lambda_recon: float = 10.0
    recon_num_bins: int = 32
    recon_sigma_factor: float = 0.5
    recon_mode: str = "js"
    use_scheduler: bool = False
    vae_init_ckpt: str = ""

    # Outputs
    ckpt_dir: str = "./workdir"
    sample_dir: str = "./workdir/sampled_cells"
    metric_dir: str = "./workdir/metrics"
    plot_dir: str = "./workdir/plots"
    feat_by_label: str = "./workdir/feat_by_label"


__all__ = ["UnifiedConfig"]
