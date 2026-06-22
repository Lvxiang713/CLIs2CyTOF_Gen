from __future__ import annotations

import argparse
import json

from singlecell_generative_unified.analysis.pipeline import AnalysisConfig, run_analysis


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(description="Unified one-click analysis runner for FM/DDPM/VAE/GAN.")
    parser.add_argument("--method", choices=["flowmatching", "ddpm", "vae", "gan"], required=True)
    parser.add_argument("--resume_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ehr_csv", default="your_ehr_csv_datapath")
    parser.add_argument("--sc_csv", default="your_single_cell_csv_datapath")
    parser.add_argument("--label_xlsx", default="your_label_xlsx_datapath")
    parser.add_argument("--clip_cfg", default="finalPara2/CLIPcheckpointDir/clip_config.json")
    parser.add_argument("--clip_ckpt", default="finalPara2/CLIPcheckpointDir/bestCLIP_model.pth")
    parser.add_argument("--notebook_source", default="GenNoiseAnalysis_Group_umapFixed_0130_addEHRmask_globalMaskCurve.ipynb")
    parser.add_argument("--sample_source", choices=["train", "val", "test", "train_val", "all", "donor_list"], default="all")
    parser.add_argument("--donor_ids", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_dims", type=int, default=36)
    parser.add_argument("--condition_mode", choices=["cls", "all_tokens", "feature_tokens"], default="feature_tokens")
    parser.add_argument("--condition_token_keep_all_max", type=float, default=None)
    parser.add_argument("--condition_token_dropout_enabled", action="store_true")
    parser.add_argument("--model_dims", type=int, default=128)
    parser.add_argument("--dims_mult", nargs="+", type=int, default=[1, 2, 2, 2, 2])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attention_resolutions", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout_attn", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--flow_method", choices=["euler", "odeint"], default="euler")
    parser.add_argument("--flow_euler_steps", type=int, default=100)
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument("--ode_solver", default="dopri5")
    parser.add_argument("--ddpm_pred_type", choices=["eps", "x0"], default="x0")
    parser.add_argument("--ddpm_sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--no_clip_denoised", action="store_true")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--beta_schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_cells_total", type=int, default=1000)
    parser.add_argument("--no_record_cross_attn", action="store_true")
    parser.add_argument("--attn_aggregate", choices=["mean", "sum", "first", "last"], default="mean")
    parser.add_argument("--ref_h5ad", default="")
    parser.add_argument("--label_key", default="cell_type_lvl2")
    parser.add_argument("--no_umap", action="store_true")
    parser.add_argument("--no_label_transfer", action="store_true")
    parser.add_argument("--no_cross_attn", action="store_true")
    parser.add_argument("--no_noise_analysis", action="store_true")
    parser.add_argument("--no_route_a", action="store_true")
    parser.add_argument("--no_mask_curve", action="store_true")
    parser.add_argument("--no_hist_eval", action="store_true")
    parser.add_argument("--no_export_npy_final", action="store_true")
    parser.add_argument("--no_reuse_artifacts", action="store_true")
    parser.add_argument("--force_resample", action="store_true")
    parser.add_argument("--no_task_checkpoints", action="store_true")
    parser.add_argument("--force_rerun_tasks", nargs="*", default=[])
    args = parser.parse_args()
    return AnalysisConfig(
        method=("gan" if args.method == "gan" else args.method),
        resume_ckpt=args.resume_ckpt,
        ehr_csv=args.ehr_csv,
        sc_csv=args.sc_csv,
        label_xlsx=args.label_xlsx,
        clip_cfg=args.clip_cfg,
        clip_ckpt=args.clip_ckpt,
        output_dir=args.output_dir,
        notebook_source=args.notebook_source,
        sample_source=args.sample_source,
        donor_ids=tuple(args.donor_ids),
        seed=args.seed,
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
        num_cells_total=args.num_cells_total,
        record_cross_attn=not args.no_record_cross_attn,
        attn_aggregate=args.attn_aggregate,
        ref_h5ad=args.ref_h5ad,
        label_key=args.label_key,
        run_umap=not args.no_umap,
        run_label_transfer=not args.no_label_transfer,
        run_cross_attn=not args.no_cross_attn,
        run_noise_analysis=not args.no_noise_analysis,
        run_route_a=not args.no_route_a,
        run_mask_curve=not args.no_mask_curve,
        run_hist_eval=not args.no_hist_eval,
        export_npy_final=not args.no_export_npy_final,
        reuse_artifacts=not args.no_reuse_artifacts,
        force_resample=args.force_resample,
        use_task_checkpoints=not args.no_task_checkpoints,
        force_rerun_tasks=tuple(args.force_rerun_tasks),
    )


def main() -> None:
    cfg = parse_args()
    summary = run_analysis(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
