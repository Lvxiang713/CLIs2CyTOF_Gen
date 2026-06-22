from __future__ import annotations

import argparse
import os
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from singlecell_generative_unified.configs.unified_config import UnifiedConfig
from singlecell_generative_unified.data.ehr_sc_loader import build_dataloaders, build_datasets, set_global_seed
from singlecell_generative_unified.evaluation.generation_eval import compare_generated_to_original, plot_feature_histogram_by_label
from singlecell_generative_unified.models.backbones.cvae import SingleCellCVAE, SingleCellDecoder
from singlecell_generative_unified.models.backbones.single_cell_an import SingleCellAN
from singlecell_generative_unified.models.backbones.gan import ViTCondDiscriminator
from singlecell_generative_unified.models.conditions.clip_condition import ClipConditionEncoder
from singlecell_generative_unified.models.methods.ddpm import DDPMMethod
from singlecell_generative_unified.models.methods.flow_matching import FlowMatchingMethod
from singlecell_generative_unified.models.methods.vae import VAEMethod
from singlecell_generative_unified.models.methods.gan import GANMethod
from singlecell_generative_unified.sampling.chunk_sampler import ChunkSampler, SamplingConfig
from singlecell_generative_unified.training.ddp_utils import cleanup_ddp, is_rank_zero, setup_ddp, unwrap_model
from singlecell_generative_unified.training.trainer import GenerativeTrainer
from singlecell_generative_unified.training.gan_trainer import GANTrainer


def parse_args() -> UnifiedConfig:
    defaults = UnifiedConfig()
    parser = argparse.ArgumentParser(description="Unified runner for Flow Matching, DDPM, VAE, and GAN.")

    parser.add_argument("--method", choices=["flowmatching", "ddpm", "vae", "gan"], default=defaults.method)
    parser.add_argument("--mode", choices=["train_sample", "sample_only"], default=defaults.mode)
    parser.add_argument("--resume_ckpt", default=defaults.resume_ckpt)
    parser.add_argument("--skip_eval", action="store_true")

    parser.add_argument("--ehr_csv", default=defaults.ehr_csv)
    parser.add_argument("--sc_csv", default=defaults.sc_csv)
    parser.add_argument("--label_xlsx", default=defaults.label_xlsx)
    parser.add_argument("--clip_cfg", default=defaults.clip_cfg)
    parser.add_argument("--clip_ckpt", default=defaults.clip_ckpt)

    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--lr_d", type=float, default=defaults.lr_d)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--early_stop", type=int, default=defaults.early_stop)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--no_compile", action="store_true")

    parser.add_argument("--train_val_limit", type=int, default=defaults.train_val_limit)
    parser.add_argument("--train_ratio", type=float, default=defaults.train_ratio)

    parser.add_argument("--feature_dims", type=int, default=defaults.feature_dims)
    parser.add_argument("--condition_mode", choices=["cls", "all_tokens", "feature_tokens"], default=defaults.condition_mode)
    parser.add_argument("--condition_token_keep_all_max", type=float, default=defaults.condition_token_keep_all_max)
    parser.add_argument("--condition_token_dropout_enabled", action="store_true")
    parser.add_argument("--model_dims", type=int, default=defaults.model_dims)
    parser.add_argument("--dims_mult", type=int, nargs="+", default=list(defaults.dims_mult))
    parser.add_argument("--num_res_blocks", type=int, default=defaults.num_res_blocks)
    parser.add_argument("--attention_resolutions", type=int, nargs="+", default=list(defaults.attention_resolutions))
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--dropout_attn", type=float, default=defaults.dropout_attn)
    parser.add_argument("--num_heads", type=int, default=defaults.num_heads)

    parser.add_argument("--scheduler_factor", type=float, default=defaults.scheduler_factor)
    parser.add_argument("--scheduler_patience", type=int, default=defaults.scheduler_patience)
    parser.add_argument("--min_lr", type=float, default=defaults.min_lr)

    parser.add_argument("--sample_source", choices=["train", "val", "test", "train_val", "all", "donor_list"], default=defaults.sample_source)
    parser.add_argument("--sample_donor_ids", nargs="*", default=list(defaults.sample_donor_ids))
    parser.add_argument("--num_cells_total", type=int, default=defaults.num_cells_total)
    parser.add_argument("--cell_num_per_sample", type=int, default=defaults.cell_num_per_sample)

    parser.add_argument("--flow_method", choices=["euler", "odeint"], default=defaults.flow_method)
    parser.add_argument("--flow_euler_steps", type=int, default=defaults.flow_euler_steps)
    parser.add_argument("--ode_rtol", type=float, default=defaults.ode_rtol)
    parser.add_argument("--ode_atol", type=float, default=defaults.ode_atol)
    parser.add_argument("--ode_solver", default=defaults.ode_solver)

    parser.add_argument("--ddpm_pred_type", choices=["eps", "x0"], default=defaults.ddpm_pred_type)
    parser.add_argument("--ddpm_sampler", choices=["ddpm", "ddim"], default=defaults.ddpm_sampler)
    parser.add_argument("--ddim_steps", type=int, default=defaults.ddim_steps)
    parser.add_argument("--eta", type=float, default=defaults.eta)
    parser.add_argument("--no_clip_denoised", action="store_true")
    parser.add_argument("--diffusion_steps", type=int, default=defaults.diffusion_steps)
    parser.add_argument("--beta_schedule", choices=["linear", "cosine"], default=defaults.beta_schedule)

    parser.add_argument("--latent_dim", type=int, default=defaults.latent_dim)
    parser.add_argument("--kl_weight", type=float, default=defaults.kl_weight)

    parser.add_argument("--n_critic", type=int, default=defaults.n_critic)
    parser.add_argument("--lambda_gp", type=float, default=defaults.lambda_gp)
    parser.add_argument("--lambda_recon", type=float, default=defaults.lambda_recon)
    parser.add_argument("--recon_num_bins", type=int, default=defaults.recon_num_bins)
    parser.add_argument("--recon_sigma_factor", type=float, default=defaults.recon_sigma_factor)
    parser.add_argument("--recon_mode", choices=["js", "kl", "l2"], default=defaults.recon_mode)
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--vae_init_ckpt", default=defaults.vae_init_ckpt)

    parser.add_argument("--ckpt_dir", default=defaults.ckpt_dir)
    parser.add_argument("--sample_dir", default=defaults.sample_dir)
    parser.add_argument("--metric_dir", default=defaults.metric_dir)
    parser.add_argument("--plot_dir", default=defaults.plot_dir)
    parser.add_argument("--feat_by_label", default=defaults.feat_by_label)

    args = parser.parse_args()
    method = 'gan' if args.method == 'gan' else args.method
    return UnifiedConfig(
        method=method,
        mode=args.mode,
        resume_ckpt=args.resume_ckpt,
        skip_eval=args.skip_eval,
        ehr_csv=args.ehr_csv,
        sc_csv=args.sc_csv,
        label_xlsx=args.label_xlsx,
        clip_cfg=args.clip_cfg,
        clip_ckpt=args.clip_ckpt,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_d=args.lr_d,
        epochs=args.epochs,
        early_stop=args.early_stop,
        seed=args.seed,
        num_workers=args.num_workers,
        compile_model=not args.no_compile,
        train_val_limit=args.train_val_limit,
        train_ratio=args.train_ratio,
        feature_dims=args.feature_dims,
        condition_mode=args.condition_mode,
        condition_token_keep_all_max=args.condition_token_keep_all_max,
        condition_token_dropout_enabled=args.condition_token_dropout_enabled,
        model_dims=args.model_dims,
        dims_mult=tuple(args.dims_mult),
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(args.attention_resolutions),
        dropout=args.dropout,
        dropout_attn=args.dropout_attn,
        num_heads=args.num_heads,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        min_lr=args.min_lr,
        sample_source=args.sample_source,
        sample_donor_ids=tuple(args.sample_donor_ids),
        num_cells_total=args.num_cells_total,
        cell_num_per_sample=args.cell_num_per_sample,
        flow_method=args.flow_method,
        flow_euler_steps=args.flow_euler_steps,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_solver=args.ode_solver,
        ddpm_pred_type=args.ddpm_pred_type,
        ddpm_sampler=args.ddpm_sampler,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
        clip_denoised=not args.no_clip_denoised,
        diffusion_steps=args.diffusion_steps,
        beta_schedule=args.beta_schedule,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        lambda_recon=args.lambda_recon,
        recon_num_bins=args.recon_num_bins,
        recon_sigma_factor=args.recon_sigma_factor,
        recon_mode=args.recon_mode,
        use_scheduler=args.use_scheduler,
        vae_init_ckpt=args.vae_init_ckpt,
        ckpt_dir=args.ckpt_dir,
        sample_dir=args.sample_dir,
        metric_dir=args.metric_dir,
        plot_dir=args.plot_dir,
        feat_by_label=args.feat_by_label,
    )


def _resolve_donor_ids(bundle, cfg: UnifiedConfig) -> list[str]:
    if cfg.sample_source == 'train':
        return list(bundle.train_donor_ids)
    if cfg.sample_source == 'val':
        return list(bundle.val_donor_ids)
    if cfg.sample_source == 'test':
        return list(bundle.test_donor_ids)
    if cfg.sample_source == 'train_val':
        seen = set(bundle.train_donor_ids)
        return list(bundle.train_donor_ids) + [d for d in bundle.val_donor_ids if d not in seen]
    if cfg.sample_source == 'all':
        seen = set()
        out = []
        for donor_id in list(bundle.train_donor_ids) + list(bundle.val_donor_ids) + list(bundle.test_donor_ids):
            if donor_id not in seen:
                seen.add(donor_id)
                out.append(donor_id)
        return out
    if cfg.sample_source == 'donor_list':
        if not cfg.sample_donor_ids:
            raise ValueError('sample_donor_ids must be provided when sample_source=donor_list')
        return list(cfg.sample_donor_ids)
    raise ValueError(f'Unsupported sample_source: {cfg.sample_source}')


def _normalize_loaded_state_dict(state_obj: object) -> OrderedDict[str, torch.Tensor]:
    if isinstance(state_obj, dict) and 'model_state_dict' in state_obj:
        state_obj = state_obj['model_state_dict']
    if not isinstance(state_obj, dict):
        raise TypeError('Checkpoint must be a state dict or contain model_state_dict.')
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_obj.items():
        new_key = key
        while True:
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
                continue
            if new_key.startswith('_orig_mod.'):
                new_key = new_key[len('_orig_mod.'):]
                continue
            break
        normalized[new_key] = value
    return normalized


def _make_sampling_cfg(cfg: UnifiedConfig) -> SamplingConfig:
    if cfg.method == 'flowmatching':
        return SamplingConfig(
            num_cells_total=cfg.num_cells_total,
            cell_num_per_sample=cfg.cell_num_per_sample,
            feature_num=cfg.feature_dims,
            output_dir=cfg.sample_dir,
            time_log_path=os.path.join(cfg.ckpt_dir, f'sampling_time_log_flow_{cfg.flow_method}.csv'),
            sampler=cfg.flow_method,
            euler_steps=cfg.flow_euler_steps,
            ode_rtol=cfg.ode_rtol,
            ode_atol=cfg.ode_atol,
            ode_solver=cfg.ode_solver,
        )
    if cfg.method == 'ddpm':
        return SamplingConfig(
            num_cells_total=cfg.num_cells_total,
            cell_num_per_sample=cfg.cell_num_per_sample,
            feature_num=cfg.feature_dims,
            output_dir=cfg.sample_dir,
            time_log_path=os.path.join(cfg.ckpt_dir, f'sampling_time_log_ddpm_{cfg.ddpm_sampler}.csv'),
            sampler=cfg.ddpm_sampler,
            euler_steps=cfg.ddim_steps,
        )
    return SamplingConfig(
        num_cells_total=cfg.num_cells_total,
        cell_num_per_sample=cfg.cell_num_per_sample,
        feature_num=cfg.feature_dims,
        output_dir=cfg.sample_dir,
        time_log_path=os.path.join(cfg.ckpt_dir, f'sampling_time_log_{cfg.method}.csv'),
        sampler=cfg.method,
    )


def _load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    state_obj = torch.load(ckpt_path, map_location=device)
    normalized = _normalize_loaded_state_dict(state_obj)
    unwrap_model(model).load_state_dict(normalized, strict=False)


def _build_score_backbone(cfg: UnifiedConfig, cond_dim: int, cond_seq_len: int) -> torch.nn.Module:
    return SingleCellAN(
        feature_dims=cfg.feature_dims,
        EHR_embdims=cond_dim,
        EHR_fdims=cond_seq_len,
        model_dims=cfg.model_dims,
        dims_mult=cfg.dims_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=cfg.dropout,
        dropout_attn=cfg.dropout_attn,
        num_heads=cfg.num_heads,
        condition_token_keep_all_max=cfg.condition_token_keep_all_max,
        condition_token_dropout_enabled=cfg.condition_token_dropout_enabled,
    )


def _build_vae_backbone(cfg: UnifiedConfig, cond_dim: int, cond_seq_len: int) -> torch.nn.Module:
    return SingleCellCVAE(
        feature_dims=cfg.feature_dims,
        EHR_embdims=cond_dim,
        EHR_fdims=cond_seq_len,
        model_dims=cfg.model_dims,
        dims_mult=cfg.dims_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=cfg.dropout,
        dropoutAtt=cfg.dropout_attn,
        num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dim,
        condition_token_keep_all_max=cfg.condition_token_keep_all_max,
        condition_token_dropout_enabled=cfg.condition_token_dropout_enabled,
    )


def _build_gan_generator(cfg: UnifiedConfig, cond_dim: int, cond_seq_len: int) -> torch.nn.Module:
    return SingleCellDecoder(
        feature_dims=cfg.feature_dims,
        EHR_embdims=cond_dim,
        EHR_fdims=cond_seq_len,
        model_dims=cfg.model_dims,
        dims_mult=cfg.dims_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=cfg.dropout,
        dropoutAtt=cfg.dropout_attn,
        num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dim,
        condition_token_keep_all_max=cfg.condition_token_keep_all_max,
        condition_token_dropout_enabled=cfg.condition_token_dropout_enabled,
    )


def ddp_worker(rank: int, world_size: int, cfg: UnifiedConfig) -> None:
    use_ddp = world_size > 1
    if use_ddp:
        setup_ddp(rank, world_size)
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cpu")
        
    # setup_ddp(rank, world_size)
    # device = torch.device(f'cuda:{rank}')
    set_global_seed(cfg.seed + rank)

    condition_encoder = ClipConditionEncoder(
        cfg.clip_cfg,
        cfg.clip_ckpt,
        device,
        condition_mode=cfg.condition_mode,
    )
    bundle = build_datasets(cfg.ehr_csv, cfg.sc_csv, cfg.train_val_limit, cfg.train_ratio, cfg.seed)
    train_loader, val_loader, test_loader = build_dataloaders(bundle, cfg.batch_size, world_size, rank, cfg.num_workers)
    donor_ids_to_sample = _resolve_donor_ids(bundle, cfg)
    donor_condition_dict = condition_encoder.build_donor_condition_dict(bundle.dataset, donor_ids_to_sample)
    sampler = ChunkSampler(device=device)

    if cfg.method in {'flowmatching', 'ddpm'}:
        model = _build_score_backbone(cfg, condition_encoder.condition_dim, condition_encoder.condition_seq_len).to(device)
        if cfg.mode != 'sample_only' and cfg.compile_model:
            model = torch.compile(model)
        if use_ddp:
            model = DDP(model, device_ids=[rank])
        if cfg.method == 'flowmatching':
            method = FlowMatchingMethod(time_log_path=os.path.join(cfg.ckpt_dir, 'flow_matching_train_time.log'))
        else:
            method = DDPMMethod(timesteps=cfg.diffusion_steps, beta_schedule=cfg.beta_schedule, pred_type=cfg.ddpm_pred_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler_factor, patience=cfg.scheduler_patience, min_lr=cfg.min_lr)
        trainer = GenerativeTrainer(device=device, ckpt_dir=cfg.ckpt_dir, num_epochs=cfg.epochs, early_stop=cfg.early_stop)
        if cfg.mode == 'train_sample':
            trainer.fit(model=model, method=method, condition_encoder=condition_encoder, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler)
        else:
            if not cfg.resume_ckpt:
                raise ValueError('resume_ckpt must be provided for sample_only mode.')
            _load_ckpt_into_model(model, cfg.resume_ckpt, device)
            if is_rank_zero():
                print(f'Loaded checkpoint for sample_only: {cfg.resume_ckpt}')
        if is_rank_zero():
            sampler.sample_by_donor(model=unwrap_model(model), method=method, donor_condition_dict=donor_condition_dict, donor_ids=donor_ids_to_sample, cfg=_make_sampling_cfg(cfg))

    elif cfg.method == 'vae':
        model = _build_vae_backbone(cfg, condition_encoder.condition_dim, condition_encoder.condition_seq_len).to(device)
        if use_ddp:
            model = DDP(model, device_ids=[rank])
        method = VAEMethod(kl_weight=cfg.kl_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler_factor, patience=cfg.scheduler_patience, min_lr=cfg.min_lr)
        trainer = GenerativeTrainer(device=device, ckpt_dir=cfg.ckpt_dir, num_epochs=cfg.epochs, early_stop=cfg.early_stop)
        if cfg.mode == 'train_sample':
            trainer.fit(model=model, method=method, condition_encoder=condition_encoder, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler)
        else:
            if not cfg.resume_ckpt:
                raise ValueError('resume_ckpt must be provided for sample_only mode.')
            _load_ckpt_into_model(model, cfg.resume_ckpt, device)
            if is_rank_zero():
                print(f'Loaded checkpoint for sample_only: {cfg.resume_ckpt}')
        if is_rank_zero():
            sampler.sample_by_donor(model=unwrap_model(model), method=method, donor_condition_dict=donor_condition_dict, donor_ids=donor_ids_to_sample, cfg=_make_sampling_cfg(cfg))

    elif cfg.method == 'gan':
        generator = _build_gan_generator(cfg, condition_encoder.condition_dim, condition_encoder.condition_seq_len).to(device)
        discriminator = ViTCondDiscriminator(
            numFeas=cfg.feature_dims,
            cond_dim=condition_encoder.condition_dim,
            vit_embed_dim=128,
        ).to(device)
        
        if use_ddp:
            generator = DDP(generator, device_ids=[rank])
            discriminator = DDP(discriminator, device_ids=[rank])
        method = GANMethod(latent_dim=cfg.latent_dim)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_d, betas=(0.0, 0.9))
        scheduler_g = None
        if cfg.use_scheduler:
            scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=cfg.scheduler_factor, patience=cfg.scheduler_patience, min_lr=cfg.min_lr)
        trainer = GANTrainer(
            device=device,
            ckpt_dir=cfg.ckpt_dir,
            num_epochs=cfg.epochs,
            latent_dim=cfg.latent_dim,
            feature_dims=cfg.feature_dims,
            n_critic=cfg.n_critic,
            lambda_gp=cfg.lambda_gp,
            lambda_recon=cfg.lambda_recon,
            recon_num_bins=cfg.recon_num_bins,
            recon_sigma_factor=cfg.recon_sigma_factor,
            recon_mode=cfg.recon_mode,
        )
        if cfg.mode == 'train_sample':
            trainer.fit(generator=generator, discriminator=discriminator, condition_encoder=condition_encoder, train_loader=train_loader, val_loader=val_loader, optimizer_g=optimizer_g, optimizer_d=optimizer_d, scheduler_g=scheduler_g, vae_init_ckpt=cfg.vae_init_ckpt)
        else:
            if not cfg.resume_ckpt:
                raise ValueError('resume_ckpt must be provided for sample_only mode.')
            _load_ckpt_into_model(generator, cfg.resume_ckpt, device)
            if is_rank_zero():
                print(f'Loaded checkpoint for sample_only: {cfg.resume_ckpt}')
        if is_rank_zero():
            sampler.sample_by_donor(model=unwrap_model(generator), method=method, donor_condition_dict=donor_condition_dict, donor_ids=donor_ids_to_sample, cfg=_make_sampling_cfg(cfg))
    else:
        raise ValueError(f'Unsupported method: {cfg.method}')

    if is_rank_zero() and not cfg.skip_eval:
        compare_generated_to_original(cfg.sc_csv, cfg.sample_dir, cfg.metric_dir, cfg.plot_dir, cfg.feature_dims)
        plot_feature_histogram_by_label(cfg.sample_dir, cfg.sc_csv, cfg.label_xlsx, donor_ids_to_sample, cfg.feat_by_label, cfg.feature_dims)

    if use_ddp:
        cleanup_ddp()

def main():
    cfg = parse_args()

    if cfg.mode == "sample_only":
        # In sample-only mode, run a single process on one visible GPU.
        ddp_worker(rank=0, world_size=1, cfg=cfg)
    else:
        world_size = torch.cuda.device_count()
        mp.spawn(ddp_worker, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
