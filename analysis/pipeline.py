from __future__ import annotations

import inspect
import json
import os
import random
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from scipy.linalg import det, inv, sqrtm

from singlecell_generative_unified.analysis.artifact_sampling import (
    make_cond_from_donor,
    make_cond_from_ehr_array,
    sample_ddpm_artifact,
    sample_ddpm_final_from_given_noise,
    sample_flowmatching_euler_artifact,
    sample_flowmatching_final_from_given_x0,
    sample_flowmatching_odeint_artifact,
    sample_one_step_artifact,
    sample_one_step_final_from_given_latent,
    save_donor_artifact,
)
from singlecell_generative_unified.analysis.legacy_notebook_loader import load_legacy_notebook_namespace
from singlecell_generative_unified.configs.unified_config import UnifiedConfig
from singlecell_generative_unified.data.ehr_sc_loader import build_datasets, set_global_seed
from singlecell_generative_unified.evaluation.generation_eval import compare_generated_to_original, plot_feature_histogram_by_label
from singlecell_generative_unified.models.conditions.clip_condition import ClipConditionEncoder
from singlecell_generative_unified.models.methods.ddpm import DDPMMethod
from singlecell_generative_unified.models.methods.flow_matching import FlowMatchingMethod
from singlecell_generative_unified.models.methods.vae import VAEMethod
from singlecell_generative_unified.models.methods.gan import GANMethod
from singlecell_generative_unified.run import _build_score_backbone, _build_vae_backbone, _build_gan_generator, _load_ckpt_into_model


@dataclass
class AnalysisConfig:
    method: str = "flowmatching"
    resume_ckpt: str = ""
    ehr_csv: str = "your_ehr_csv_datapath"
    sc_csv: str = "your_single_cell_csv_datapath"
    label_xlsx: str = "your_label_xlsx_datapath"
    clip_cfg: str = "finalPara2/CLIPcheckpointDir/clip_config.json"
    clip_ckpt: str = "finalPara2/CLIPcheckpointDir/bestCLIP_model.pth"
    output_dir: str = "./analysis_out"
    notebook_source: str = "GenNoiseAnalysis_Group_umapFixed_0130_addEHRmask_globalMaskCurve.ipynb"
    sample_source: str = "all"
    donor_ids: tuple[str, ...] = field(default_factory=tuple)
    seed: int = 42
    feature_dims: int = 36
    condition_mode: str = "feature_tokens"
    condition_token_keep_all_max: float | None = None
    condition_token_dropout_enabled: bool = False
    model_dims: int = 128
    dims_mult: tuple[int, ...] = (1, 2, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (2, 4, 8, 16)
    dropout: float = 0.0
    dropout_attn: float = 0.1
    num_heads: int = 4
    flow_method: str = "euler"
    flow_euler_steps: int = 100
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-5
    ode_solver: str = "dopri5"
    ddpm_pred_type: str = "x0"
    ddpm_sampler: str = "ddpm"
    ddim_steps: int = 50
    eta: float = 0.0
    clip_denoised: bool = True
    diffusion_steps: int = 1000
    beta_schedule: str = "linear"
    latent_dim: int = 128
    kl_weight: float = 0.1
    num_cells_total: int = 1000
    record_cross_attn: bool = True
    attn_aggregate: str = "mean"
    ref_h5ad: str = ""
    label_key: str = "cell_type_lvl2"
    run_umap: bool = True
    run_label_transfer: bool = True
    run_cross_attn: bool = True
    run_noise_analysis: bool = True
    run_route_a: bool = True
    run_mask_curve: bool = True
    run_hist_eval: bool = True
    export_npy_final: bool = True
    reuse_artifacts: bool = True
    force_resample: bool = False
    use_task_checkpoints: bool = True
    force_rerun_tasks: tuple[str, ...] = field(default_factory=tuple)
    route_a_clip_k_list: tuple[float, ...] = (3.0, 2.0, 1.0, 0.5, 0.1, 0.0)
    route_a_min_count: int = 1
    feature_min_cells_for_arrow: int = 1
    feature_min_cells_for_heatmap: int = 1
    mask_k_list: tuple[int, ...] = (1, 3, 5, 10)
    mask_random_repeats: int = 3
    mask_modes: tuple[str, ...] = ("token_drop",)

    def to_unified(self) -> UnifiedConfig:
        method = "gan" if self.method == "gan" else self.method
        return UnifiedConfig(
            method=method,
            mode="sample_only",
            resume_ckpt=self.resume_ckpt,
            ehr_csv=self.ehr_csv,
            sc_csv=self.sc_csv,
            label_xlsx=self.label_xlsx,
            clip_cfg=self.clip_cfg,
            clip_ckpt=self.clip_ckpt,
            seed=self.seed,
            feature_dims=self.feature_dims,
            condition_mode=self.condition_mode,
            condition_token_keep_all_max=self.condition_token_keep_all_max,
            condition_token_dropout_enabled=self.condition_token_dropout_enabled,
            model_dims=self.model_dims,
            dims_mult=self.dims_mult,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=self.attention_resolutions,
            dropout=self.dropout,
            dropout_attn=self.dropout_attn,
            num_heads=self.num_heads,
            sample_source=self.sample_source,
            sample_donor_ids=self.donor_ids,
            num_cells_total=self.num_cells_total,
            cell_num_per_sample=self.num_cells_total,
            flow_method=self.flow_method,
            flow_euler_steps=self.flow_euler_steps,
            ode_rtol=self.ode_rtol,
            ode_atol=self.ode_atol,
            ode_solver=self.ode_solver,
            ddpm_pred_type=self.ddpm_pred_type,
            ddpm_sampler=self.ddpm_sampler,
            ddim_steps=self.ddim_steps,
            eta=self.eta,
            clip_denoised=self.clip_denoised,
            diffusion_steps=self.diffusion_steps,
            beta_schedule=self.beta_schedule,
            latent_dim=self.latent_dim,
            kl_weight=self.kl_weight,
        )


def _resolve_donor_ids(bundle, cfg: UnifiedConfig) -> list[str]:
    dataset = bundle.dataset
    all_donors = sorted(list(dataset.ehr_dict.keys()))
    if cfg.sample_source == "all":
        return all_donors
    if cfg.sample_source == "donor_list":
        return list(cfg.sample_donor_ids)
    if cfg.sample_source == "train":
        return list(bundle.train_donor_ids)
    if cfg.sample_source == "val":
        return list(bundle.val_donor_ids)
    if cfg.sample_source == "test":
        return list(bundle.test_donor_ids)
    if cfg.sample_source == "train_val":
        out = list(bundle.train_donor_ids) + list(bundle.val_donor_ids)
        return list(dict.fromkeys(out))
    raise ValueError(f"Unknown sample_source={cfg.sample_source}")


def _build_model_and_method(cfg: UnifiedConfig, condition_encoder: ClipConditionEncoder, device: torch.device):
    method_name = "gan" if cfg.method == "gan" else cfg.method
    cond_dim = condition_encoder.condition_dim
    cond_seq_len = condition_encoder.condition_seq_len
    if method_name in {"flowmatching", "ddpm"}:
        model = _build_score_backbone(cfg, cond_dim, cond_seq_len).to(device)
        method = FlowMatchingMethod() if method_name == "flowmatching" else DDPMMethod(timesteps=cfg.diffusion_steps, beta_schedule=cfg.beta_schedule, pred_type=cfg.ddpm_pred_type)
    elif method_name == "vae":
        model = _build_vae_backbone(cfg, cond_dim, cond_seq_len).to(device)
        method = VAEMethod(kl_weight=cfg.kl_weight)
    elif method_name == "gan":
        model = _build_gan_generator(cfg, cond_dim, cond_seq_len).to(device)
        method = GANMethod(latent_dim=cfg.latent_dim)
    else:
        raise ValueError(f"Unsupported method={cfg.method}")
    if not cfg.resume_ckpt:
        raise ValueError("resume_ckpt is required for analysis.")
    _load_ckpt_into_model(model, cfg.resume_ckpt, device)
    model.eval()
    return model, method


def _artifact_path(artifact_dir: Path, donor_id: str) -> Path:
    return artifact_dir / f"{donor_id}.pt"


def _artifact_is_usable(path: Path, cfg: AnalysisConfig, method_name: str) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        obj = _torch_load_compat(path, map_location="cpu")
    except Exception as exc:
        return False, f"load_failed: {exc}"
    traj = obj.get("traj", None)
    if not isinstance(traj, torch.Tensor) or traj.dim() != 3:
        return False, "bad_traj"
    if int(traj.shape[1]) != int(cfg.num_cells_total) or int(traj.shape[2]) != int(cfg.feature_dims):
        return False, f"shape_mismatch: got={tuple(traj.shape)} expected=(*,{cfg.num_cells_total},{cfg.feature_dims})"
    analysis_mode = obj.get("analysis_mode", None)
    expected_mode = "one_step" if method_name in {"vae", "gan"} else "multistep"
    if analysis_mode is not None and analysis_mode != expected_mode:
        return False, f"analysis_mode_mismatch: {analysis_mode} != {expected_mode}"
    if cfg.record_cross_attn and cfg.run_cross_attn and obj.get("cross_attn", None) is None:
        return False, "missing_cross_attn"
    return True, "ok"


def _plan_artifact_reuse(artifact_dir: Path, donor_ids: list[str], cfg: AnalysisConfig, method_name: str) -> tuple[list[str], list[str], dict[str, str]]:
    reused: list[str] = []
    to_sample: list[str] = []
    reasons: dict[str, str] = {}
    for donor_id in donor_ids:
        path = _artifact_path(artifact_dir, donor_id)
        if cfg.force_resample or not cfg.reuse_artifacts:
            to_sample.append(donor_id)
            reasons[donor_id] = "force_resample" if cfg.force_resample else "reuse_disabled"
            continue
        ok, reason = _artifact_is_usable(path, cfg, method_name)
        if ok:
            reused.append(donor_id)
        else:
            to_sample.append(donor_id)
            reasons[donor_id] = reason
    return reused, to_sample, reasons


def _export_final_npy(artifact_dir: Path, npy_dir: Path) -> None:
    npy_dir.mkdir(parents=True, exist_ok=True)
    for pt_path in sorted(artifact_dir.glob("*.pt")):
        obj = _torch_load_compat(pt_path, map_location="cpu")
        traj = obj["traj"]
        final = traj[-1].numpy().astype(np.float32)
        np.save(npy_dir / f"{pt_path.stem}.npy", final)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _task_state_path(out_root: Path) -> Path:
    return out_root / "analysis_task_state.json"


def _load_task_state(out_root: Path) -> dict[str, Any]:
    path = _task_state_path(out_root)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_task_state(out_root: Path, state: dict[str, Any]) -> None:
    _save_json(_task_state_path(out_root), state)


def _task_force_rerun(cfg: AnalysisConfig, task_name: str) -> bool:
    return str(task_name) in set(str(x) for x in cfg.force_rerun_tasks)


def _should_run_task(cfg: AnalysisConfig, state: dict[str, Any], task_name: str, outputs_ok) -> bool:
    if not cfg.use_task_checkpoints:
        return True
    if _task_force_rerun(cfg, task_name):
        return True
    entry = state.get(task_name, None)
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return True
    try:
        return not bool(outputs_ok())
    except Exception:
        return True


def _mark_task_state(out_root: Path, state: dict[str, Any], task_name: str, payload: dict[str, Any]) -> None:
    state[str(task_name)] = payload
    _save_task_state(out_root, state)


def _cache_dir(out_dir: Path) -> Path:
    d = out_dir / "_task_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _task_ok_payload(outputs: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"status": "ok"}
    if outputs:
        payload["outputs"] = outputs
    return payload


def _task_fail_payload(exc: Exception | str) -> dict[str, Any]:
    return {"status": "failed", "error": str(exc)}


def _torch_load_compat(path: str | os.PathLike[str], map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@contextmanager
def _legacy_torch_load_context():
    orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        yield
    finally:
        torch.load = orig_load


def _legacy_globals_dict(legacy) -> dict[str, object]:
    for name in dir(legacy):
        if name.startswith("__"):
            continue
        obj = getattr(legacy, name)
        if callable(obj) and hasattr(obj, "__globals__"):
            return obj.__globals__
    return {}


def _inject_legacy_globals(legacy, **values: object) -> None:
    updated = set()
    for name in dir(legacy):
        if name.startswith("__"):
            continue
        obj = getattr(legacy, name)
        if callable(obj) and hasattr(obj, "__globals__"):
            g = obj.__globals__
            gid = id(g)
            if gid in updated:
                continue
            for k, v in values.items():
                g[k] = v
            updated.add(gid)
    for k, v in values.items():
        setattr(legacy, k, v)


def _read_h5ad_compat(legacy, ref_h5ad: str, feature_names: list[str]):
    read_sig = inspect.signature(legacy.read_h5ad_X)
    read_kwargs: dict[str, Any] = {}
    if "layer" in read_sig.parameters:
        read_kwargs["layer"] = None
    elif "use_layer" in read_sig.parameters:
        read_kwargs["use_layer"] = None
    if "use_raw" in read_sig.parameters:
        read_kwargs["use_raw"] = False
    read_out = legacy.read_h5ad_X(ref_h5ad, **read_kwargs)
    if not isinstance(read_out, tuple):
        raise TypeError("legacy.read_h5ad_X must return a tuple")
    if len(read_out) == 3:
        X_ref, ref_feature_names, adata = read_out
    elif len(read_out) == 2:
        X_ref, ref_feature_names = read_out
        adata = None
    else:
        raise TypeError(f"Unexpected legacy.read_h5ad_X return length: {len(read_out)}")
    X_ref, ref_feature_names = legacy.align_by_feature_names(X_ref, ref_feature_names, feature_names)
    return X_ref, ref_feature_names, adata


def _load_all_donor_objs(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pt_path in sorted(artifact_dir.glob("*.pt")):
        obj = _torch_load_compat(pt_path, map_location="cpu")
        out[str(obj.get("donor_id", pt_path.stem))] = obj
    return out


def _prepare_feature_names(legacy, sc_csv: str, dims: int) -> list[str]:
    if hasattr(legacy, "get_feature_names_from_sc_csv"):
        try:
            names = legacy.get_feature_names_from_sc_csv(sc_csv)
            if isinstance(names, list) and len(names) >= dims:
                return list(names[:dims])
        except Exception:
            pass
    df = pd.read_csv(sc_csv, nrows=1)
    cols = [c for c in df.columns if c != "sample"]
    if len(cols) < dims:
        raise ValueError(f"sc_csv has only {len(cols)} feature columns, expected at least {dims}")
    return cols[:dims]


def _prepare_ehr_feature_names(dataset, ehr_csv: str) -> list[str]:
    if hasattr(dataset, "ehr_feature_names"):
        names = getattr(dataset, "ehr_feature_names")
        if isinstance(names, (list, tuple)) and len(names) > 0:
            return list(names)
    df = pd.read_csv(ehr_csv, nrows=1)
    cols = [c for c in df.columns if c not in {"donor_ID", "donor_id", "donor", "id"}]
    return cols


def _run_trace_stats(legacy, artifact_dir: Path, out_dir: Path) -> None:
    with _legacy_torch_load_context():
        ok_ids, all_std, all_mse, updates = legacy.load_all_donors_std_mse_from_pt_dir(str(artifact_dir), strict_steps=False)
    overall_sig = inspect.signature(legacy.plot_overall_std_mse)
    overall_kwargs = {"title_prefix": f"All donors (N={len(ok_ids)})"} if "title_prefix" in overall_sig.parameters else {}
    fig1 = legacy.plot_overall_std_mse(updates, all_std, all_mse, **overall_kwargs)
    fig1.savefig(out_dir / "overall_std_mse.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)
    overlay_sig = inspect.signature(legacy.plot_overlay_std_mse)
    overlay_kwargs = {"title_prefix": f"All donors (N={len(ok_ids)})"} if "title_prefix" in overlay_sig.parameters else {}
    fig2 = legacy.plot_overlay_std_mse(updates, all_std, all_mse, **overlay_kwargs)
    fig2.savefig(out_dir / "overlay_std_mse.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

def _safe_file_stem(x: Any) -> str:
    s = str(x)
    s = "".join(c if (c.isalnum() or c in ("_", "-", ".")) else "_" for c in s)
    return s[:180] if len(s) > 180 else s


def _save_feature_steps_heatmap_raw_values(
    *,
    artifact_dir: Path,
    final_label_map: dict[str, np.ndarray],
    feature_names: list[str],
    out_dir: Path,
) -> None:
    """
    Save raw feature heatmap values without filtering any cell type.

    It computes mean, std, var, and count for each:
        step × final cell type × feature

    Output:
        feature_steps_heatmaps/raw_values/
            step_000_feature_mean_by_celltype.csv
            step_000_feature_std_by_celltype.csv
            step_000_feature_var_by_celltype.csv
            step_000_celltype_counts.csv

            step_000_feature_mean_by_celltype.npy
            step_000_feature_std_by_celltype.npy
            step_000_feature_var_by_celltype.npy

            all_steps_feature_mean_std_by_celltype.npz
    """
    raw_dir = out_dir / "raw_values"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sums: dict[tuple[int, str], np.ndarray] = {}
    sumsqs: dict[tuple[int, str], np.ndarray] = {}
    counts: dict[tuple[int, str], int] = {}

    all_labels: set[str] = set()
    step_indices: set[int] = set()
    n_features_ref: int | None = None

    skipped_rows: list[dict[str, Any]] = []

    pt_files = sorted(artifact_dir.glob("*.pt"))
    if not pt_files:
        print(f"[feature_raw_values] No .pt files found in {artifact_dir}")
        return

    for pt_path in pt_files:
        obj = _torch_load_compat(pt_path, map_location="cpu")

        donor_id = str(obj.get("donor_id", pt_path.stem))

        labels = final_label_map.get(donor_id, None)
        if labels is None:
            skipped_rows.append({
                "pt_file": pt_path.name,
                "donor_id": donor_id,
                "reason": "no_labels_in_final_label_map",
            })
            continue

        traj = obj.get("traj", None)
        if traj is None:
            skipped_rows.append({
                "pt_file": pt_path.name,
                "donor_id": donor_id,
                "reason": "no_traj",
            })
            continue

        if torch.is_tensor(traj):
            arr = traj.detach().cpu().float().numpy()
        else:
            arr = np.asarray(traj, dtype=np.float32)

        # Compatible with possible shape:
        # steps × 1 × cells × features
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]

        # Expected shape:
        # steps × cells × features
        if arr.ndim != 3:
            skipped_rows.append({
                "pt_file": pt_path.name,
                "donor_id": donor_id,
                "reason": f"unsupported_traj_shape_{arr.shape}",
            })
            continue

        n_steps, n_cells, n_features = arr.shape

        labels = np.asarray(labels, dtype=object).astype(str)

        if labels.shape[0] != n_cells:
            skipped_rows.append({
                "pt_file": pt_path.name,
                "donor_id": donor_id,
                "reason": f"label_cell_mismatch_labels_{labels.shape[0]}_cells_{n_cells}",
            })
            continue

        if n_features_ref is None:
            n_features_ref = n_features
        elif n_features != n_features_ref:
            skipped_rows.append({
                "pt_file": pt_path.name,
                "donor_id": donor_id,
                "reason": f"feature_dim_mismatch_features_{n_features}_ref_{n_features_ref}",
            })
            continue

        arr = arr.astype(np.float64)

        for step_idx in range(n_steps):
            step_indices.add(step_idx)
            X = arr[step_idx]

            for label in np.unique(labels):
                label = str(label)
                mask = labels == label
                n = int(mask.sum())

                if n <= 0:
                    continue

                all_labels.add(label)

                key = (step_idx, label)

                if key not in sums:
                    sums[key] = np.zeros((n_features,), dtype=np.float64)
                    sumsqs[key] = np.zeros((n_features,), dtype=np.float64)
                    counts[key] = 0

                Xk = X[mask]

                sums[key] += Xk.sum(axis=0)
                sumsqs[key] += np.square(Xk).sum(axis=0)
                counts[key] += n

    if n_features_ref is None or not all_labels or not step_indices:
        print("[feature_raw_values] No valid data to save.")
        if skipped_rows:
            pd.DataFrame(skipped_rows).to_csv(
                raw_dir / "skipped_artifacts_when_computing_feature_raw_values.csv",
                index=False,
            )
        return

    labels_sorted = sorted(all_labels)
    step_sorted = sorted(step_indices)

    if len(feature_names) == n_features_ref:
        cols = list(feature_names)
    else:
        cols = [f"feature_{i}" for i in range(n_features_ref)]

    all_mean_mats = []
    all_std_mats = []
    all_var_mats = []
    all_count_mats = []

    for step_idx in step_sorted:
        mean_mat = np.full(
            (len(labels_sorted), n_features_ref),
            np.nan,
            dtype=np.float32,
        )

        var_mat = np.full(
            (len(labels_sorted), n_features_ref),
            np.nan,
            dtype=np.float32,
        )

        std_mat = np.full(
            (len(labels_sorted), n_features_ref),
            np.nan,
            dtype=np.float32,
        )

        count_vec = np.zeros(
            (len(labels_sorted),),
            dtype=np.int64,
        )

        for i, label in enumerate(labels_sorted):
            key = (step_idx, label)

            if key not in counts or counts[key] <= 0:
                continue

            n = counts[key]
            mean = sums[key] / n

            # Population variance, same convention as np.var(axis=0).
            var = sumsqs[key] / n - np.square(mean)
            var = np.maximum(var, 0.0)
            std = np.sqrt(var)

            mean_mat[i] = mean.astype(np.float32)
            var_mat[i] = var.astype(np.float32)
            std_mat[i] = std.astype(np.float32)
            count_vec[i] = int(n)

        mean_df = pd.DataFrame(mean_mat, index=labels_sorted, columns=cols)
        mean_df.index.name = "cell_type"
        mean_df.to_csv(raw_dir / f"step_{step_idx:03d}_feature_mean_by_celltype.csv")

        std_df = pd.DataFrame(std_mat, index=labels_sorted, columns=cols)
        std_df.index.name = "cell_type"
        std_df.to_csv(raw_dir / f"step_{step_idx:03d}_feature_std_by_celltype.csv")

        var_df = pd.DataFrame(var_mat, index=labels_sorted, columns=cols)
        var_df.index.name = "cell_type"
        var_df.to_csv(raw_dir / f"step_{step_idx:03d}_feature_var_by_celltype.csv")

        count_df = pd.DataFrame({
            "cell_type": labels_sorted,
            "count": count_vec,
        })
        count_df.to_csv(
            raw_dir / f"step_{step_idx:03d}_celltype_counts.csv",
            index=False,
        )

        np.save(
            raw_dir / f"step_{step_idx:03d}_feature_mean_by_celltype.npy",
            mean_mat,
        )

        np.save(
            raw_dir / f"step_{step_idx:03d}_feature_std_by_celltype.npy",
            std_mat,
        )

        np.save(
            raw_dir / f"step_{step_idx:03d}_feature_var_by_celltype.npy",
            var_mat,
        )

        all_mean_mats.append(mean_mat)
        all_std_mats.append(std_mat)
        all_var_mats.append(var_mat)
        all_count_mats.append(count_vec)

    np.savez_compressed(
        raw_dir / "all_steps_feature_mean_std_by_celltype.npz",
        mean=np.stack(all_mean_mats, axis=0),
        std=np.stack(all_std_mats, axis=0),
        var=np.stack(all_var_mats, axis=0),
        counts=np.stack(all_count_mats, axis=0),
        step_indices=np.asarray(step_sorted, dtype=np.int64),
        labels=np.asarray(labels_sorted, dtype=object),
        feature_names=np.asarray(cols, dtype=object),
    )

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(
            raw_dir / "skipped_artifacts_when_computing_feature_raw_values.csv",
            index=False,
        )

    print("[feature_raw_values] Saved mean/std/var/count raw values to:", raw_dir)

def _save_attn_mean_by_type_raw_values(
    *,
    attn_mean_by_type: Any,
    meta: Any,
    out_dir: Path,
) -> None:
    """
    Save raw attention matrices before plotting.

    Output:
        cross_attn_steps_heatmaps/raw_values/
            attn_mean_by_type_raw.joblib
            attn_raw_summary.csv
            beta.npy
            beta.csv
            epsilon.npy
            epsilon_step_000.csv
            ...
    """
    raw_dir = out_dir / "raw_values"
    raw_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "attn_mean_by_type": attn_mean_by_type,
            "meta": meta,
        },
        raw_dir / "attn_mean_by_type_raw.joblib",
    )

    if not isinstance(attn_mean_by_type, dict):
        return

    summary_rows = []

    for label, value in sorted(attn_mean_by_type.items(), key=lambda x: str(x[0])):
        label_str = str(label)
        stem = _safe_file_stem(label_str)

        try:
            if torch.is_tensor(value):
                arr = value.detach().cpu().float().numpy()
            else:
                arr = np.asarray(value)
        except Exception as exc:
            summary_rows.append({
                "label": label_str,
                "shape": "unavailable",
                "saved": False,
                "reason": str(exc),
            })
            continue

        np.save(raw_dir / f"{stem}.npy", arr)

        nan_count = 0
        if np.issubdtype(arr.dtype, np.number):
            nan_count = int(np.isnan(arr).sum())

        summary_rows.append({
            "label": label_str,
            "shape": str(tuple(arr.shape)),
            "dtype": str(arr.dtype),
            "nan_count": nan_count,
            "saved": True,
        })

        if arr.ndim == 1:
            pd.DataFrame({
                "index": np.arange(arr.shape[0]),
                "value": arr,
            }).to_csv(raw_dir / f"{stem}.csv", index=False)

        elif arr.ndim == 2:
            pd.DataFrame(arr).to_csv(raw_dir / f"{stem}.csv", index=False)

        elif arr.ndim == 3:
            for step_idx in range(arr.shape[0]):
                pd.DataFrame(arr[step_idx]).to_csv(
                    raw_dir / f"{stem}_step_{step_idx:03d}.csv",
                    index=False,
                )

        elif arr.ndim == 4:
            for step_idx in range(arr.shape[0]):
                flat = arr[step_idx].reshape(arr.shape[1], -1)
                pd.DataFrame(flat).to_csv(
                    raw_dir / f"{stem}_step_{step_idx:03d}_flatten.csv",
                    index=False,
                )

    pd.DataFrame(summary_rows).to_csv(raw_dir / "attn_raw_summary.csv", index=False)
def _run_umap_and_label_transfer(legacy, cfg: AnalysisConfig, artifact_dir: Path, out_dir: Path, feature_names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not cfg.ref_h5ad or not os.path.exists(cfg.ref_h5ad):
        result["status"] = "skipped_no_ref_h5ad"
        return result

    X_ref, ref_feature_names, adata = _read_h5ad_compat(legacy, cfg.ref_h5ad, feature_names)
    bundle_dir = out_dir / "label_transfer_umaplearn_bundle"

    if cfg.run_label_transfer:
        if not legacy.bundle_exists(str(bundle_dir)):
            if adata is not None:
                labels = legacy.get_obs_labels(adata, cfg.label_key)
            elif hasattr(legacy, "read_h5ad_X_and_labels"):
                xyl = legacy.read_h5ad_X_and_labels(
                    cfg.ref_h5ad,
                    label_key=cfg.label_key,
                    use_layer=None,
                    use_raw=False,
                    ref_feature_names=feature_names,
                )
                _, labels, _, _ = xyl
            else:
                raise RuntimeError("Cannot recover labels from reference h5ad with current legacy notebook functions.")

            legacy.build_label_transfer_bundle(
                bundle_dir=str(bundle_dir),
                X_ref=X_ref,
                y_ref=labels,
                var_names=ref_feature_names,
                label_key=cfg.label_key,
                ref_h5ad=cfg.ref_h5ad,
                n_pca=min(30, X_ref.shape[1]),
                knn_k=30,
                umap_kwargs=None,
            )

        pca, knn, umap_model, meta = legacy.load_label_transfer_bundle(str(bundle_dir))

        with _legacy_torch_load_context():
            donor_objs = legacy.load_all_donor_objs(str(artifact_dir))
            X_final, obs_final, final_step_idx = legacy.build_query_from_pt_final_step(
                pt_dir=str(artifact_dir),
                dims_expected=len(feature_names),
                donor_ids=None,
                final_step_idx=None,
            )

        pred_final, conf, proba, Y_final = legacy.predict_labels_and_umap(X_final, pca, knn, umap_model)

        pred_df = obs_final.copy()
        pred_df["pred_label"] = np.asarray(pred_final, dtype=object)
        if conf is not None:
            pred_df["confidence"] = np.asarray(conf, dtype=np.float32)
        pred_df["umap1"] = np.asarray(Y_final[:, 0], dtype=np.float32)
        pred_df["umap2"] = np.asarray(Y_final[:, 1], dtype=np.float32)
        pred_df.to_csv(out_dir / "predicted_labels_umap.csv", index=False)

        final_label_map = legacy.build_final_label_map_from_pred(obs_final, pred_final, label_col="pred_label")

        result.update({
            "final_label_map": final_label_map,
            "donor_objs": donor_objs,
            "final_step_idx": int(final_step_idx),
            "pca": pca,
            "knn": knn,
            "umap_model": umap_model,
            "bundle_meta": meta,
            "X_ref": X_ref,
            "ref_feature_names": ref_feature_names,
        })

        joblib.dump(
            {
                "pca": pca,
                "knn": knn,
                "umap_model": umap_model,
                "meta": meta,
            },
            out_dir / "label_transfer_bundle.joblib",
        )

        cache_dir = _cache_dir(out_dir)
        joblib.dump(final_label_map, cache_dir / "final_label_map.joblib")

        if cfg.run_umap:
            Z_ref = pca.transform(X_ref)
            Y_ref = umap_model.transform(Z_ref)
            xlim, ylim = legacy.compute_fixed_umap_limits(Y_ref)

            with _legacy_torch_load_context():
                legacy.render_steps_centroid_arrows_and_heatmaps(
                    PT_DIR=str(artifact_dir),
                    pca=pca,
                    umap_model=umap_model,
                    donor_objs=donor_objs,
                    final_label_map=final_label_map,
                    feature_names=feature_names,
                    xlim=xlim,
                    ylim=ylim,
                    n_plots=10,
                    show_arrows=True,
                    min_cells_for_arrow=1,
                    head_scale=0.01,
                    min_cells_for_heatmap=1,
                    topk_types_heatmap=None,
                    point_size=0.2,
                    point_alpha=0.05,
                    save_dir_umap=str(out_dir / "umap_steps_centroid_arrows"),
                    save_dir_feat=str(out_dir / "feature_steps_heatmaps"),
                    legend=True,
                )

            _save_feature_steps_heatmap_raw_values(
                artifact_dir=artifact_dir,
                final_label_map=final_label_map,
                feature_names=feature_names,
                out_dir=out_dir / "feature_steps_heatmaps",
            )

            result["feature_steps_heatmaps_raw_values"] = str(out_dir / "feature_steps_heatmaps" / "raw_values")

    result["status"] = "ok"
    return result

def _load_cached_label_transfer_context(legacy, cfg: AnalysisConfig, artifact_dir: Path, out_dir: Path, feature_names: list[str]) -> dict[str, Any]:
    cache_dir = _cache_dir(out_dir)
    final_map_path = cache_dir / "final_label_map.joblib"
    bundle_dir = out_dir / "label_transfer_umaplearn_bundle"
    if not final_map_path.exists() or not bundle_dir.exists():
        raise FileNotFoundError("Missing cached label transfer artifacts.")
    final_label_map = joblib.load(final_map_path)
    pca, knn, umap_model, meta = legacy.load_label_transfer_bundle(str(bundle_dir))
    donor_objs = _load_all_donor_objs(artifact_dir)
    X_ref = None
    ref_feature_names = None
    if cfg.ref_h5ad and os.path.exists(cfg.ref_h5ad):
        X_ref, ref_feature_names, _ = _read_h5ad_compat(legacy, cfg.ref_h5ad, feature_names)
    final_step_idx = None
    pred_csv = out_dir / "predicted_labels_umap.csv"
    if pred_csv.exists():
        try:
            pred_df = pd.read_csv(pred_csv)
            if "step_idx" in pred_df.columns and len(pred_df) > 0:
                final_step_idx = int(pred_df["step_idx"].iloc[0])
        except Exception:
            pass
    return {
        "status": "ok",
        "final_label_map": final_label_map,
        "donor_objs": donor_objs,
        "final_step_idx": final_step_idx,
        "pca": pca,
        "knn": knn,
        "umap_model": umap_model,
        "bundle_meta": meta,
        "X_ref": X_ref,
        "ref_feature_names": ref_feature_names,
    }

def _safe_copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)


def _diagnosis_map_from_xlsx(label_xlsx: str) -> dict[str, str]:
    if not label_xlsx or not os.path.exists(label_xlsx):
        return {}
    try:
        df = pd.read_excel(label_xlsx)
    except Exception:
        return {}
    donor_col = None
    dx_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if donor_col is None and cl in {"donor_id", "donorid"}:
            donor_col = c
        if dx_col is None and cl in {"clinical_diagnosis", "clinicaldiagnosis", "diagnosis"}:
            dx_col = c
    if donor_col is None or dx_col is None:
        return {}

    def _norm_dx(x: str) -> str:
        x = str(x).strip()
        xl = x.lower()
        if xl in ["control", "t1d control", "t1d_control"]:
            return "Control"
        if xl in ["t1d", "type1", "type 1", "type1diabetes", "type 1 diabetes"]:
            return "T1D"
        return x

    return {str(did): _norm_dx(dx) for did, dx in zip(df[donor_col].astype(str), df[dx_col].astype(str))}


def _build_final_label_map_dx(final_label_map: dict[str, np.ndarray], donor_to_dx: dict[str, str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for did, labels in final_label_map.items():
        g = donor_to_dx.get(str(did), "Unknown")
        arr = np.asarray(labels, dtype=object)
        out[str(did)] = np.asarray([f"{g}|{ct}" for ct in arr], dtype=object)
    return out

def _run_cross_attention_exports(
    legacy,
    artifact_dir: Path,
    out_dir: Path,
    final_label_map: dict[str, np.ndarray],
    label_xlsx: str,
    donor_objs: dict[str, Any] | None,
    feature_names: list[str],
    ehr_feature_names: list[str],
) -> dict[str, Any]:
    staging = out_dir / "_crossattn_staging"
    staging.mkdir(parents=True, exist_ok=True)

    for pt in artifact_dir.glob("*.pt"):
        link = staging / pt.name
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            os.symlink(pt.resolve(), link)
        except Exception:
            shutil.copy2(pt, link)

    _inject_legacy_globals(
        legacy,
        PT_DIR=str(staging),
        donor_objs=donor_objs,
        final_label_map=final_label_map,
        feature_names=feature_names,
        ref_feature_names=feature_names,
        ehr_feature_names=ehr_feature_names,
    )

    outputs: dict[str, Any] = {}

    with _legacy_torch_load_context():
        attn_mean_by_type, meta = legacy.aggregate_cross_attn_by_celltype_from_pt_dir(
            pt_dir=str(staging),
            final_label_map=final_label_map,
            donor_objs=donor_objs,
            include_all=True,
            fill_step0="copy_step1",
            verbose=True,
        )

        legacy.save_attn_steps_heatmaps_like_feature_steps(
            attn_mean_by_type=attn_mean_by_type,
            out_dir=str(staging / "cross_attn_steps_heatmaps"),
            meta=meta,
            prefix="CrossAttnSteps",
            dpi=300,
            figsize=(10, 6),
            fontname="Arial",
            label_fs=18,
            tick_fs=12,
            save_csv=True,
            save_npy=True,
        )

    _save_attn_mean_by_type_raw_values(
        attn_mean_by_type=attn_mean_by_type,
        meta=meta,
        out_dir=staging / "cross_attn_steps_heatmaps",
    )

    outputs["cross_attn_steps_heatmaps"] = str(out_dir / "cross_attn_steps_heatmaps")
    outputs["cross_attn_steps_heatmaps_raw_values"] = str(out_dir / "cross_attn_steps_heatmaps" / "raw_values")

    donor_to_dx = _diagnosis_map_from_xlsx(label_xlsx)
    final_label_map_dx = None

    if donor_to_dx:
        final_label_map_dx = _build_final_label_map_dx(final_label_map, donor_to_dx)

        with _legacy_torch_load_context():
            attn_mean_by_type_dx, meta_dx = legacy.aggregate_cross_attn_by_celltype_from_pt_dir(
                pt_dir=str(staging),
                final_label_map=final_label_map_dx,
                donor_objs=donor_objs,
                include_all=True,
                fill_step0="copy_step1",
                verbose=True,
            )

            legacy.save_attn_steps_heatmaps_like_feature_steps(
                attn_mean_by_type=attn_mean_by_type_dx,
                out_dir=str(staging / "cross_attn_steps_heatmaps_by_dx"),
                meta=meta_dx,
                prefix="CrossAttnSteps",
                dpi=300,
                figsize=(10, 6),
                fontname="Arial",
                label_fs=18,
                tick_fs=12,
                save_csv=True,
                save_npy=True,
            )

        _save_attn_mean_by_type_raw_values(
            attn_mean_by_type=attn_mean_by_type_dx,
            meta=meta_dx,
            out_dir=staging / "cross_attn_steps_heatmaps_by_dx",
        )

        outputs["cross_attn_steps_heatmaps_by_dx"] = str(out_dir / "cross_attn_steps_heatmaps_by_dx")
        outputs["cross_attn_steps_heatmaps_by_dx_raw_values"] = str(out_dir / "cross_attn_steps_heatmaps_by_dx" / "raw_values")

    with _legacy_torch_load_context():
        legacy.build_importance_tensors(final_label_map, "cross_attn_importance_full")

    outputs["cross_attn_importance_full"] = str(out_dir / "cross_attn_importance_full")

    if final_label_map_dx is not None:
        with _legacy_torch_load_context():
            legacy.build_importance_tensors(final_label_map_dx, "cross_attn_importance_full_by_dx")

        outputs["cross_attn_importance_full_by_dx"] = str(out_dir / "cross_attn_importance_full_by_dx")

    if hasattr(legacy, "export_ehr_feature_importance_and_heatmaps"):
        with _legacy_torch_load_context():
            legacy.export_ehr_feature_importance_and_heatmaps(
                final_label_map,
                "cross_attn_ehr_feature_importance",
                reduce_steps="mean",
                step_idx=-1,
                reduce_protein="mean",
                top_k_plot=30,
                skip_if_exists=False,
            )

        outputs["cross_attn_ehr_feature_importance"] = str(out_dir / "cross_attn_ehr_feature_importance")

        if final_label_map_dx is not None:
            with _legacy_torch_load_context():
                legacy.export_ehr_feature_importance_and_heatmaps(
                    final_label_map_dx,
                    "cross_attn_ehr_feature_importance_by_dx",
                    reduce_steps="mean",
                    step_idx=-1,
                    reduce_protein="mean",
                    top_k_plot=30,
                    skip_if_exists=False,
                )

            outputs["cross_attn_ehr_feature_importance_by_dx"] = str(out_dir / "cross_attn_ehr_feature_importance_by_dx")

    if hasattr(legacy, "make_all_45_plots"):
        with _legacy_torch_load_context():
            legacy.make_all_45_plots(str(staging / "cross_attn_importance_full"), "raw_celltype")

        outputs["cross_attn_45_plots_raw"] = str(out_dir / "cross_attn_importance_full" / "plots_45_raw_celltype")

        if final_label_map_dx is not None and (staging / "cross_attn_importance_full_by_dx").exists():
            with _legacy_torch_load_context():
                legacy.make_all_45_plots(str(staging / "cross_attn_importance_full_by_dx"), "dx_celltype")

            outputs["cross_attn_45_plots_by_dx"] = str(out_dir / "cross_attn_importance_full_by_dx" / "plots_45_dx_celltype")

    for name in [
        "cross_attn_steps_heatmaps",
        "cross_attn_steps_heatmaps_by_dx",
        "cross_attn_importance_full",
        "cross_attn_importance_full_by_dx",
        "cross_attn_ehr_feature_importance",
        "cross_attn_ehr_feature_importance_by_dx",
    ]:
        src = staging / name
        dst = out_dir / name
        if src.exists():
            _safe_copytree(src, dst)

    with open(out_dir / "cross_attn_summary.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    return outputs

def _run_one_step_noise_summary(artifact_dir: Path, out_dir: Path) -> None:
    rows = []
    for pt_path in sorted(artifact_dir.glob("*.pt")):
        obj = _torch_load_compat(pt_path, map_location="cpu")
        z = obj.get("latent_noise", None)
        if z is None:
            continue
        z_np = z.numpy().astype(np.float32)
        rows.append({"donor_id": pt_path.stem, "mean": float(z_np.mean()), "std": float(z_np.std()), "min": float(z_np.min()), "max": float(z_np.max())})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "latent_noise_summary.csv", index=False)
    plt.figure(figsize=(8, 4))
    plt.hist(df["mean"], bins=20, alpha=0.7, label="mean")
    plt.hist(df["std"], bins=20, alpha=0.7, label="std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "latent_noise_summary.png", dpi=200, bbox_inches="tight")
    plt.close()


def _predict_labels_with_bundle(X: np.ndarray, pca, knn) -> tuple[np.ndarray, np.ndarray]:
    Z = pca.transform(X)
    pred = knn.predict(Z).astype(object)
    if hasattr(knn, "predict_proba"):
        try:
            proba = knn.predict_proba(Z)
            conf = proba.max(axis=1).astype(np.float32)
        except Exception:
            conf = np.full((len(pred),), np.nan, dtype=np.float32)
    else:
        conf = np.full((len(pred),), np.nan, dtype=np.float32)
    return pred, conf

def _build_diag_stats_from_artifacts(artifact_dir: Path, final_label_map: dict[str, np.ndarray], key: str, min_count: int) -> dict[str, dict[str, Any]]:
    X_list = []
    y_list = []

    for pt_path in sorted(artifact_dir.glob("*.pt")):
        obj = _torch_load_compat(pt_path, map_location="cpu")
        did = str(obj.get("donor_id", pt_path.stem))
        labels = final_label_map.get(did, None)

        if labels is None:
            continue

        if key == "data_step0":
            X = obj["traj"][0].cpu().numpy().astype(np.float32)
        elif key == "latent_noise":
            z = obj.get("latent_noise", None)
            if z is None:
                continue
            X = z.cpu().numpy().astype(np.float32)
        else:
            raise ValueError(key)

        labels = np.asarray(labels, dtype=object)

        if labels.shape[0] != X.shape[0]:
            continue

        X_list.append(X)
        y_list.append(labels)

    if not X_list:
        return {}

    X0 = np.vstack(X_list)
    y0 = np.concatenate(y_list)

    labels, counts = np.unique(y0, return_counts=True)

    # no filtering: keep every label that appears at least once
    keep = labels

    stats: dict[str, dict[str, Any]] = {}

    for k in keep.tolist():
        Xk = X0[y0 == k]
        if Xk.shape[0] <= 0:
            continue

        stats[str(k)] = {
            "count": int(Xk.shape[0]),
            "mu": Xk.mean(axis=0).astype(np.float32),
            "var": (Xk.var(axis=0) + 1e-8).astype(np.float32),
        }

    return stats

def _sample_diag_gaussian(stats: dict[str, dict[str, Any]], label: str, n_cells: int, seed: int, clip_k: float | None) -> np.ndarray:
    s = stats[str(label)]
    rng = np.random.default_rng(int(seed))
    mu = s["mu"].astype(np.float32)
    std = np.sqrt(s["var"]).astype(np.float32)
    eps = rng.standard_normal(size=(n_cells, mu.shape[0]), dtype=np.float32)
    x0 = mu[None, :] + eps * std[None, :]
    if clip_k is not None:
        lo = mu[None, :] - float(clip_k) * std[None, :]
        hi = mu[None, :] + float(clip_k) * std[None, :]
        x0 = np.clip(x0, lo, hi)
    return x0.astype(np.float32)


def _clip_tag(clip_k: float | None) -> str:
    if clip_k is None:
        return "noclip"
    s = f"{float(clip_k):.3f}".rstrip("0").rstrip(".")
    return f"clip{s.replace('.', 'p')}sigma"


def _generate_from_initial(
    *,
    unified: UnifiedConfig,
    model: torch.nn.Module,
    method_obj,
    cond: tuple[torch.Tensor, torch.Tensor | None],
    init_arr: np.ndarray,
    cfg: AnalysisConfig,
) -> np.ndarray:
    if unified.method == "flowmatching":
        return sample_flowmatching_final_from_given_x0(model=model, cond=cond, x0=init_arr, steps=cfg.flow_euler_steps)
    if unified.method == "ddpm":
        return sample_ddpm_final_from_given_noise(model=model, method=method_obj, cond=cond, x_init=init_arr, sampler=cfg.ddpm_sampler, ddim_steps=cfg.ddim_steps, eta=cfg.eta, clip_denoised=cfg.clip_denoised)
    if unified.method in {"vae", "gan"}:
        target_model = model if unified.method == "gan" else model.decoder
        return sample_one_step_final_from_given_latent(model=target_model, cond=cond, z_init=init_arr)
    raise ValueError(unified.method)


def _run_route_a(
    *,
    legacy,
    cfg: AnalysisConfig,
    unified: UnifiedConfig,
    artifact_dir: Path,
    out_dir: Path,
    donor_ids: list[str],
    bundle,
    condition_encoder: ClipConditionEncoder,
    model: torch.nn.Module,
    method_obj,
    final_label_map: dict[str, np.ndarray] | None,
    pca,
    knn,
    feature_names: list[str],
) -> dict[str, Any]:
    if final_label_map is None or pca is None or knn is None:
        return {"status": "skipped_no_label_transfer_bundle"}
    key = "latent_noise" if unified.method in {"vae", "gan"} else "data_step0"
    stats = _build_diag_stats_from_artifacts(artifact_dir, final_label_map, key=key, min_count=cfg.route_a_min_count)
    if not stats:
        return {"status": "skipped_no_diag_stats"}
    base_dir = out_dir / "noise_label_sampling_A_all_donors"
    base_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = base_dir / "diag_gaussian"
    diag_dir.mkdir(parents=True, exist_ok=True)
    labels_sorted = sorted(stats.keys())
    mu_mat = np.stack([stats[k]["mu"] for k in labels_sorted], axis=0)
    var_mat = np.stack([stats[k]["var"] for k in labels_sorted], axis=0)
    cnt_vec = np.array([stats[k]["count"] for k in labels_sorted], dtype=np.int64)
    np.savez_compressed(diag_dir / f"diag_gaussian_stats_{key}.npz", labels=np.array(labels_sorted, dtype=object), mu=mu_mat, var=var_mat, count=cnt_vec)

    results: dict[str, Any] = {}
    clip_values: list[float | None] = [None] + [float(x) for x in cfg.route_a_clip_k_list]
    device = next(model.parameters()).device
    for clip_k in clip_values:
        clip_tag = _clip_tag(clip_k)
        exp_dir = out_dir / f"noise_label_sampling_A_all_donors_{clip_tag}"
        pred_dir = exp_dir / "predictions"
        tab_dir = exp_dir / "tables"
        pred_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)
        done_files = [tab_dir / "purity_long_table.csv", tab_dir / "purity_delta_conditioned_minus_baseline.csv"]
        if cfg.reuse_artifacts and all(p.exists() for p in done_files):
            results[clip_tag] = {"status": "reused", "dir": str(exp_dir)}
            continue

        all_rows: list[dict[str, Any]] = []
        D = len(next(iter(stats.values()))["mu"])
        for di, donor_id in enumerate(donor_ids):
            cond = make_cond_from_donor(donor_id, dataset=bundle.dataset, condition_encoder=condition_encoder, device=device)
            rng = np.random.default_rng(int(cfg.seed) + 100000 + di)
            x0_base = rng.standard_normal(size=(cfg.num_cells_total, D), dtype=np.float32)
            X_base = _generate_from_initial(unified=unified, model=model, method_obj=method_obj, cond=cond, init_arr=x0_base, cfg=cfg)
            pred_base, conf_base = _predict_labels_with_bundle(X_base, pca, knn)
            pd.DataFrame({"donor_id": donor_id, "method": "baseline", "pred_label": pred_base, "conf": conf_base, "clip_tag": clip_tag}).to_csv(pred_dir / f"pred_baseline_donor{donor_id}.csv", index=False)
            for li, label in enumerate(labels_sorted):
                x0_cond = _sample_diag_gaussian(stats, label, cfg.num_cells_total, int(cfg.seed) + di * 1000 + li, clip_k)
                X_cond = _generate_from_initial(unified=unified, model=model, method_obj=method_obj, cond=cond, init_arr=x0_cond, cfg=cfg)
                pred_cond, conf_cond = _predict_labels_with_bundle(X_cond, pca, knn)
                pd.DataFrame({"donor_id": donor_id, "method": "conditioned", "target_label": str(label), "pred_label": pred_cond, "conf": conf_cond, "clip_tag": clip_tag}).to_csv(pred_dir / f"pred_conditioned_donor{donor_id}_target{label}.csv", index=False)
                purity_cond = float(np.mean(pred_cond == label))
                purity_base = float(np.mean(pred_base == label))
                all_rows.append({"donor_id": donor_id, "method": "conditioned", "target_label": str(label), "n_cells": int(cfg.num_cells_total), "purity": purity_cond, "mean_conf": float(np.nanmean(conf_cond)), "median_conf": float(np.nanmedian(conf_cond)), "clip_tag": clip_tag})
                all_rows.append({"donor_id": donor_id, "method": "baseline", "target_label": str(label), "n_cells": int(cfg.num_cells_total), "purity": purity_base, "mean_conf": float(np.nanmean(conf_base)), "median_conf": float(np.nanmedian(conf_base)), "clip_tag": clip_tag})
        df_long = pd.DataFrame(all_rows)
        df_long.to_csv(tab_dir / "purity_long_table.csv", index=False)
        df_cond = df_long[df_long["method"] == "conditioned"].pivot(index="donor_id", columns="target_label", values="purity")
        df_base = df_long[df_long["method"] == "baseline"].pivot(index="donor_id", columns="target_label", values="purity")
        df_cond.to_csv(tab_dir / "purity_matrix_conditioned.csv")
        df_base.to_csv(tab_dir / "purity_matrix_baseline.csv")
        (df_cond - df_base).to_csv(tab_dir / "purity_delta_conditioned_minus_baseline.csv")
        results[clip_tag] = {"status": "ok", "dir": str(exp_dir)}
    return results


def _evaluate_metrics_all_only(X_real: np.ndarray, X_gen: np.ndarray, sample_size: int = 2000, mmd_sample_size: int = 10000, mmd_gamma: float | None = None) -> dict[str, float]:
    Xr = np.asarray(X_real, dtype=np.float32)
    Xg = np.asarray(X_gen, dtype=np.float32)
    if Xr.shape[0] < 2 or Xg.shape[0] < 2:
        return dict(KL=np.nan, Wasserstein=np.nan, MMD=np.nan, Err1NN=np.nan, CD=np.nan)

    def kl_div(mu0, cov0, mu1, cov1):
        k = mu0.size
        inv1 = inv(cov1)
        diff = mu1 - mu0
        val = 0.5 * (np.trace(inv1 @ cov0) + diff.T @ inv1 @ diff - k + np.log(det(cov1) / det(cov0)))
        return float(max(val, 0.0))

    def wass_dist(mu0, cov0, mu1, cov1):
        diff = mu0 - mu1
        term1 = diff @ diff
        c1s = sqrtm(cov1)
        prod = c1s @ cov0 @ c1s
        ps = sqrtm(prod)
        term2 = np.trace(cov0 + cov1 - 2 * ps)
        return float(np.sqrt(np.real(term1 + term2)))

    def mmd_unbiased_sampled(X, Y):
        if len(X) > mmd_sample_size:
            X = X[np.random.choice(len(X), mmd_sample_size, replace=False)]
        if len(Y) > mmd_sample_size:
            Y = Y[np.random.choice(len(Y), mmd_sample_size, replace=False)]
        Z = np.vstack([X, Y])
        gamma_val = mmd_gamma
        if gamma_val is None:
            d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
            med = np.median(d2)
            gamma_val = 1.0 / (2.0 * med) if med > 0 else 1.0
        Kxx = rbf_kernel(X, X, gamma=gamma_val)
        Kyy = rbf_kernel(Y, Y, gamma=gamma_val)
        Kxy = rbf_kernel(X, Y, gamma=gamma_val)
        m, n = len(X), len(Y)
        val = (Kxx.sum() - m) / (m * (m - 1) + 1e-12) + (Kyy.sum() - n) / (n * (n - 1) + 1e-12) - 2 * Kxy.mean()
        return float(max(val, 0.0))

    def err_1nn_sampled(X, Y):
        if len(X) > sample_size:
            X = X[np.random.choice(len(X), sample_size, replace=False)]
        if len(Y) > sample_size:
            Y = Y[np.random.choice(len(Y), sample_size, replace=False)]
        Z = np.vstack([X, Y])
        labels = np.array([0] * len(X) + [1] * len(Y))
        clf = KNeighborsClassifier(n_neighbors=1)
        acc = cross_val_score(clf, Z, labels, cv=LeaveOneOut(), scoring="accuracy").mean()
        return float(1 - acc)

    def corr_discrepancy(X, Y):
        if X.shape[0] < 3 or Y.shape[0] < 3:
            return np.nan
        corr_r = np.corrcoef(X, rowvar=False)
        corr_g = np.corrcoef(Y, rowvar=False)
        corr_r = np.nan_to_num(corr_r, nan=0.0)
        corr_g = np.nan_to_num(corr_g, nan=0.0)
        idx = np.triu_indices(corr_r.shape[0], k=1)
        diff = np.abs(corr_r[idx] - corr_g[idx])
        return float(diff.mean())

    mu_r, mu_g = Xr.mean(0), Xg.mean(0)
    cov_r = np.cov(Xr, rowvar=False) + np.eye(Xr.shape[1]) * 1e-6
    cov_g = np.cov(Xg, rowvar=False) + np.eye(Xg.shape[1]) * 1e-6
    return dict(KL=kl_div(mu_r, cov_r, mu_g, cov_g), Wasserstein=wass_dist(mu_r, cov_r, mu_g, cov_g), MMD=mmd_unbiased_sampled(Xr, Xg), Err1NN=err_1nn_sampled(Xr, Xg), CD=corr_discrepancy(Xr, Xg))


def _build_mask_rank_lists(legacy, importance_path: str, current_token_labels: list[str]) -> tuple[list[int], dict[int, str]]:
    imp_df = legacy.load_importance_table(importance_path, sheet_name="All")
    rank_indices, idx_to_label, _ = legacy.build_rank_lists_from_importance(imp_df, current_token_labels)
    return rank_indices, idx_to_label


def _apply_mask_vec(ehr_vec: np.ndarray, idx_to_mask: list[int], mode: str, fill_values: np.ndarray | None = None) -> np.ndarray:
    v = np.array(ehr_vec, copy=True)
    if not idx_to_mask:
        return v
    mode = str(mode).lower()
    if mode == "token_drop":
        v[idx_to_mask] = np.nan
        return v
    if mode == "neutralize":
        if fill_values is None:
            v[idx_to_mask] = 0.0
        else:
            v[idx_to_mask] = fill_values[idx_to_mask]
        return v
    raise ValueError(f"Unknown mode: {mode}")


def _sample_final_random(
    *,
    unified: UnifiedConfig,
    model: torch.nn.Module,
    method_obj,
    cond: tuple[torch.Tensor, torch.Tensor | None],
    cfg: AnalysisConfig,
) -> np.ndarray:
    device = next(model.parameters()).device
    if unified.method == "flowmatching":
        x0 = np.random.standard_normal(size=(cfg.num_cells_total, cfg.feature_dims)).astype(np.float32)
        return sample_flowmatching_final_from_given_x0(model=model, cond=cond, x0=x0, steps=cfg.flow_euler_steps)
    if unified.method == "ddpm":
        x0 = np.random.standard_normal(size=(cfg.num_cells_total, cfg.feature_dims)).astype(np.float32)
        return sample_ddpm_final_from_given_noise(model=model, method=method_obj, cond=cond, x_init=x0, sampler=cfg.ddpm_sampler, ddim_steps=cfg.ddim_steps, eta=cfg.eta, clip_denoised=cfg.clip_denoised)
    if unified.method in {"vae", "gan"}:
        z0 = np.random.standard_normal(size=(cfg.num_cells_total, cfg.latent_dim)).astype(np.float32)
        target_model = model if unified.method == "gan" else model.decoder
        return sample_one_step_final_from_given_latent(model=target_model, cond=cond, z_init=z0)
    raise ValueError(unified.method)


def _run_mask_curve(
    *,
    legacy,
    cfg: AnalysisConfig,
    unified: UnifiedConfig,
    out_dir: Path,
    artifact_dir: Path,
    bundle,
    condition_encoder: ClipConditionEncoder,
    model: torch.nn.Module,
    method_obj,
    donor_ids: list[str],
    X_ref: np.ndarray | None,
    current_token_labels: list[str],
    importance_path: str | None,
) -> dict[str, Any]:
    if X_ref is None:
        return {"status": "skipped_no_ref_h5ad"}
    if not importance_path or not os.path.exists(importance_path):
        return {"status": "skipped_no_importance_file"}
    rank_indices, idx_to_label = _build_mask_rank_lists(legacy, importance_path, current_token_labels)
    if not rank_indices:
        return {"status": "skipped_empty_rank_indices"}
    exp_root = out_dir / "ehr_mask_exps"
    exp_root.mkdir(parents=True, exist_ok=True)
    metrics_csv = exp_root / "global_ehr_mask_curve_metrics.csv"

    baseline_objs = []
    for pt_path in sorted(artifact_dir.glob("*.pt")):
        obj = _torch_load_compat(pt_path, map_location="cpu")
        baseline_objs.append(obj["traj"][-1].cpu().numpy().astype(np.float32))
    X_base = np.vstack(baseline_objs)
    rows: list[dict[str, Any]] = []
    base_metrics = _evaluate_metrics_all_only(X_ref, X_base)
    rows.append({"Scheme": "baseline_nomask_from_pt", "MaskType": "nomask", "K": 0, "Repeat": 0, "Mode": "nomask", **base_metrics})

    fill_values = None
    if "neutralize" in cfg.mask_modes:
        X_ehr = np.vstack([np.asarray(bundle.dataset.ehr_dict[did]) for did in donor_ids])
        fill_values = np.nanmean(X_ehr, axis=0)

    for mode in cfg.mask_modes:
        for k in cfg.mask_k_list:
            schemes = [
                ("topk", 0, rank_indices[:k]),
                ("bottomk", 0, rank_indices[-k:]),
            ]
            for rep in range(cfg.mask_random_repeats):
                rng = random.Random(cfg.seed + 1000 + rep * 100 + k)
                sample_idx = rank_indices.copy()
                rng.shuffle(sample_idx)
                schemes.append(("randomk", rep, sample_idx[:k]))
            for mask_type, rep, idx_to_mask in schemes:
                scheme = f"mask_{mask_type}_k{k}_rep{rep}_{mode}"
                scheme_dir = exp_root / scheme
                donors_dir = scheme_dir / "donors"
                donors_dir.mkdir(parents=True, exist_ok=True)
                X_list = []
                for i, did in enumerate(donor_ids):
                    out_pt = donors_dir / f"{did}.pt"
                    if cfg.reuse_artifacts and out_pt.exists():
                        X_final = _torch_load_compat(out_pt, map_location="cpu")["X_final"].cpu().numpy().astype(np.float32)
                    else:
                        ehr0 = np.asarray(bundle.dataset.ehr_dict[did], dtype=np.float32)
                        ehr1 = _apply_mask_vec(ehr0, list(idx_to_mask), mode, fill_values)
                        cond = make_cond_from_ehr_array(ehr1, condition_encoder=condition_encoder, device=next(model.parameters()).device)
                        np.random.seed(cfg.seed + i)
                        torch.manual_seed(cfg.seed + i)
                        X_final = _sample_final_random(unified=unified, model=model, method_obj=method_obj, cond=cond, cfg=cfg)
                        torch.save({"donor_id": did, "X_final": torch.from_numpy(X_final), "scheme_name": scheme, "idx_to_mask": list(map(int, idx_to_mask)), "mode": mode}, out_pt)
                    X_list.append(X_final)
                X_gen = np.vstack(X_list)
                metrics = _evaluate_metrics_all_only(X_ref, X_gen)
                rows.append({"Scheme": scheme, "MaskType": mask_type, "K": int(k), "Repeat": int(rep), "Mode": mode, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(metrics_csv, index=False)
    plot_dir = exp_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["KL", "Wasserstein", "MMD", "Err1NN", "CD"]
    baseline_mask = (df["MaskType"].astype(str).str.lower() == "nomask") | (df["Scheme"].astype(str) == "baseline_nomask_from_pt")
    df_base = df[baseline_mask].copy()
    baseline_vals = {m: float(pd.to_numeric(df_base[m], errors="coerce").mean()) for m in metrics} if not df_base.empty else None
    modes = sorted(df["Mode"].astype(str).unique().tolist())
    for mode in modes:
        d = df[df["Mode"].astype(str) == str(mode)].copy()
        for metric in metrics:
            plt.figure(figsize=(6, 4))
            for mask_type in ["topk", "bottomk", "randomk"]:
                sub = d[d["MaskType"].astype(str) == mask_type].copy()
                if sub.empty:
                    continue
                if mask_type == "randomk":
                    g = sub.groupby("K")[metric].agg(["mean", "std"]).reset_index().sort_values("K")
                    plt.errorbar(g["K"], g["mean"], yerr=g["std"], fmt="-o", capsize=3, label=f"{mask_type} (mean±std)")
                else:
                    g = sub.groupby("K")[metric].mean().reset_index().sort_values("K")
                    plt.plot(g["K"], g[metric], "-o", label=mask_type)
            if baseline_vals is not None and metric in baseline_vals and np.isfinite(baseline_vals[metric]):
                plt.axhline(baseline_vals[metric], linestyle="--", linewidth=1.2, label="nomask baseline")
            plt.xlabel("k (masked feature count)")
            plt.ylabel(metric)
            plt.title(f"Global EHR Mask Curve: {metric} | mode={mode}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_dir / f"global_mask_curve_{metric}_{mode}.png", dpi=200)
            plt.close()
    return {"status": "ok", "metrics_csv": str(metrics_csv), "importance_path": str(importance_path)}


def _find_importance_file(analysis_dir: Path) -> str | None:
    cands = [
        analysis_dir / "cross_attn_ehr_feature_importance" / "overall_ehr_importance.xlsx",
        analysis_dir / "cross_attn_ehr_feature_importance" / "overall_ehr_importance.csv",
        analysis_dir / "cross_attn_ehr_feature_importance_by_dx" / "overall_ehr_importance.xlsx",
    ]
    for p in cands:
        if p.exists():
            return str(p)
    return None


def run_analysis(config: AnalysisConfig | dict[str, Any]) -> dict[str, Any]:
    cfg = config if isinstance(config, AnalysisConfig) else AnalysisConfig(**config)
    unified = cfg.to_unified()
    set_global_seed(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = Path(cfg.output_dir)
    artifact_dir = out_root / "artifacts"
    analysis_dir = out_root / "analysis"
    npy_dir = out_root / "sampled_cells_npy"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    task_state = _load_task_state(out_root)

    bundle = build_datasets(cfg.ehr_csv, cfg.sc_csv, unified.train_val_limit, unified.train_ratio, cfg.seed)
    donor_ids = _resolve_donor_ids(bundle, unified)
    reused_donor_ids, donor_ids_to_sample, reuse_reasons = _plan_artifact_reuse(artifact_dir, donor_ids, cfg, unified.method)

    def _npy_outputs_ok() -> bool:
        pts = list(artifact_dir.glob("*.pt"))
        return bool(pts) and all((npy_dir / f"{pt.stem}.npy").exists() for pt in pts)

    def _trace_outputs_ok() -> bool:
        return (analysis_dir / "overall_std_mse.png").exists() and (analysis_dir / "overlay_std_mse.png").exists()

    def _lt_outputs_ok() -> bool:
        return (analysis_dir / "predicted_labels_umap.csv").exists() and (analysis_dir / "label_transfer_umaplearn_bundle").exists() and (_cache_dir(analysis_dir) / "final_label_map.joblib").exists()

    def _cross_outputs_ok() -> bool:
        return (analysis_dir / "cross_attn_summary.json").exists()

    def _noise_outputs_ok() -> bool:
        if unified.method in {"vae", "gan"}:
            return (analysis_dir / "latent_noise_summary.csv").exists() and (analysis_dir / "latent_noise_summary.png").exists()
        entry = task_state.get("noise_analysis", {})
        return isinstance(entry, dict) and entry.get("status") == "ok"

    def _route_outputs_ok() -> bool:
        return (analysis_dir / "noise_label_sampling_A_all_donors").exists() or (analysis_dir / "noise_label_sampling_A_all_donors_latent").exists()

    def _mask_outputs_ok() -> bool:
        return (analysis_dir / "ehr_mask_exps" / "global_ehr_mask_curve_metrics.csv").exists()

    def _hist_outputs_ok() -> bool:
        return (analysis_dir / "metrics").exists() and (analysis_dir / "plots").exists()

    need_generation_stack = bool(donor_ids_to_sample)
    need_generation_stack = need_generation_stack or (cfg.run_route_a and _should_run_task(cfg, task_state, "route_a", _route_outputs_ok))
    need_generation_stack = need_generation_stack or (cfg.run_mask_curve and _should_run_task(cfg, task_state, "mask_curve", _mask_outputs_ok))

    condition_encoder = None
    model = None
    method_obj = None
    if need_generation_stack:
        condition_encoder = ClipConditionEncoder(cfg.clip_cfg, cfg.clip_ckpt, device, condition_mode=cfg.condition_mode)
        model, method_obj = _build_model_and_method(unified, condition_encoder, device)

    for donor_id in donor_ids_to_sample:
        cond = make_cond_from_donor(donor_id, dataset=bundle.dataset, condition_encoder=condition_encoder, device=device)
        cond = (cond[0].to(device), cond[1].to(device))
        if unified.method == "flowmatching":
            if cfg.flow_method == "odeint":
                artifact = sample_flowmatching_odeint_artifact(model=model, cond=cond, cell_num=cfg.num_cells_total, dims=cfg.feature_dims, intervals=10, record_cross_attn=cfg.record_cross_attn, attn_aggregate=cfg.attn_aggregate, atol=cfg.ode_atol, rtol=cfg.ode_rtol, solver=cfg.ode_solver)
            else:
                artifact = sample_flowmatching_euler_artifact(model=model, cond=cond, cell_num=cfg.num_cells_total, dims=cfg.feature_dims, steps=cfg.flow_euler_steps, record_cross_attn=cfg.record_cross_attn, attn_aggregate=cfg.attn_aggregate)
        elif unified.method == "ddpm":
            artifact = sample_ddpm_artifact(model=model, method=method_obj, cond=cond, cell_num=cfg.num_cells_total, dims=cfg.feature_dims, sampler=cfg.ddpm_sampler, ddim_steps=cfg.ddim_steps, eta=cfg.eta, clip_denoised=cfg.clip_denoised, record_cross_attn=cfg.record_cross_attn, attn_aggregate=cfg.attn_aggregate)
        elif unified.method in {"vae", "gan"}:
            artifact = sample_one_step_artifact(model=model if unified.method == "gan" else model.decoder, cond=cond, cell_num=cfg.num_cells_total, dims=cfg.feature_dims, latent_dim=cfg.latent_dim, mode=unified.method, record_cross_attn=cfg.record_cross_attn, attn_aggregate=cfg.attn_aggregate)
        else:
            raise ValueError(f"Unsupported method {unified.method}")
        save_donor_artifact(artifact_dir, donor_id, artifact)

    if cfg.export_npy_final and _should_run_task(cfg, task_state, "export_npy_final", _npy_outputs_ok):
        try:
            _export_final_npy(artifact_dir, npy_dir)
            _mark_task_state(out_root, task_state, "export_npy_final", _task_ok_payload({"npy_dir": str(npy_dir)}))
        except Exception as exc:
            _mark_task_state(out_root, task_state, "export_npy_final", _task_fail_payload(exc))
            raise

    legacy = load_legacy_notebook_namespace(cfg.notebook_source)
    feature_names = _prepare_feature_names(legacy, cfg.sc_csv, cfg.feature_dims)
    ehr_feature_names = _prepare_ehr_feature_names(bundle.dataset, cfg.ehr_csv)

    summary: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "analysis_dir": str(analysis_dir),
        "npy_dir": str(npy_dir),
        "method": unified.method,
        "num_donors": len(donor_ids),
        "reused_artifacts": reused_donor_ids,
        "resampled_artifacts": donor_ids_to_sample,
        "artifact_reuse_reasons": {k: reuse_reasons[k] for k in donor_ids_to_sample if k in reuse_reasons},
        "task_checkpoints": str(_task_state_path(out_root)),
    }
    _save_json(out_root / "analysis_config.json", asdict(cfg))

    if _should_run_task(cfg, task_state, "trace_stats", _trace_outputs_ok):
        try:
            _run_trace_stats(legacy, artifact_dir, analysis_dir)
            _mark_task_state(out_root, task_state, "trace_stats", _task_ok_payload({"overall": str(analysis_dir / "overall_std_mse.png"), "overlay": str(analysis_dir / "overlay_std_mse.png")}))
            summary["trace_stats"] = {"status": "ok"}
        except Exception as exc:
            _mark_task_state(out_root, task_state, "trace_stats", _task_fail_payload(exc))
            raise
    else:
        summary["trace_stats"] = {"status": "reused_checkpoint"}

    final_label_map = None
    donor_objs = None
    pca = knn = umap_model = X_ref = None
    if cfg.run_umap or cfg.run_label_transfer:
        if _should_run_task(cfg, task_state, "label_transfer", _lt_outputs_ok):
            try:
                lt_result = _run_umap_and_label_transfer(legacy, cfg, artifact_dir, analysis_dir, feature_names)
                summary["label_transfer"] = {k: v for k, v in lt_result.items() if k not in {"final_label_map", "donor_objs", "pca", "knn", "umap_model", "X_ref", "ref_feature_names", "bundle_meta"}}
                final_label_map = lt_result.get("final_label_map", None)
                donor_objs = lt_result.get("donor_objs", None)
                pca = lt_result.get("pca", None)
                knn = lt_result.get("knn", None)
                umap_model = lt_result.get("umap_model", None)
                X_ref = lt_result.get("X_ref", None)
                if lt_result.get("status") == "ok":
                    _mark_task_state(out_root, task_state, "label_transfer", _task_ok_payload({"predicted_labels_umap": str(analysis_dir / "predicted_labels_umap.csv")}))
                else:
                    _mark_task_state(out_root, task_state, "label_transfer", {"status": lt_result.get("status", "skipped")})
            except Exception as exc:
                _mark_task_state(out_root, task_state, "label_transfer", _task_fail_payload(exc))
                raise
        else:
            try:
                lt_result = _load_cached_label_transfer_context(legacy, cfg, artifact_dir, analysis_dir, feature_names)
                summary["label_transfer"] = {"status": "reused_checkpoint", "final_step_idx": lt_result.get("final_step_idx", None)}
                final_label_map = lt_result.get("final_label_map", None)
                donor_objs = lt_result.get("donor_objs", None)
                pca = lt_result.get("pca", None)
                knn = lt_result.get("knn", None)
                umap_model = lt_result.get("umap_model", None)
                X_ref = lt_result.get("X_ref", None)
            except Exception as exc:
                summary["label_transfer"] = {"status": f"cache_reload_failed: {exc}"}

    has_any_cross_attn = False
    for pt_path in artifact_dir.glob("*.pt"):
        obj = _torch_load_compat(pt_path, map_location="cpu")
        if obj.get("cross_attn") is not None:
            has_any_cross_attn = True
            break
    if cfg.run_cross_attn and has_any_cross_attn and final_label_map is not None:
        if _should_run_task(cfg, task_state, "cross_attn", _cross_outputs_ok):
            try:
                outputs = _run_cross_attention_exports(legacy, artifact_dir, analysis_dir, final_label_map, cfg.label_xlsx, donor_objs, feature_names, ehr_feature_names)
                summary["cross_attn"] = {"status": "ok", **outputs}
                _mark_task_state(out_root, task_state, "cross_attn", _task_ok_payload(outputs))
            except Exception as exc:
                summary["cross_attn"] = f"failed: {exc}"
                _mark_task_state(out_root, task_state, "cross_attn", _task_fail_payload(exc))
        else:
            summary["cross_attn"] = {"status": "reused_checkpoint"}

    if cfg.run_noise_analysis:
        if _should_run_task(cfg, task_state, "noise_analysis", _noise_outputs_ok):
            try:
                if unified.method in {"vae", "gan"}:
                    _run_one_step_noise_summary(artifact_dir, analysis_dir)
                    summary["noise_analysis"] = "latent_one_step"
                    _mark_task_state(out_root, task_state, "noise_analysis", _task_ok_payload({"type": "latent_one_step"}))
                else:
                    summary["noise_analysis"] = "artifacts_ready_for_legacy_multistep"
                    _mark_task_state(out_root, task_state, "noise_analysis", _task_ok_payload({"type": "artifacts_ready_for_legacy_multistep"}))
            except Exception as exc:
                _mark_task_state(out_root, task_state, "noise_analysis", _task_fail_payload(exc))
                raise
        else:
            entry = task_state.get("noise_analysis", {})
            summary["noise_analysis"] = entry.get("outputs", {}).get("type", "reused_checkpoint")

    if cfg.run_route_a:
        if _should_run_task(cfg, task_state, "route_a", _route_outputs_ok):
            try:
                if condition_encoder is None or model is None:
                    condition_encoder = ClipConditionEncoder(cfg.clip_cfg, cfg.clip_ckpt, device, condition_mode=cfg.condition_mode)
                    model, method_obj = _build_model_and_method(unified, condition_encoder, device)
                route_res = _run_route_a(legacy=legacy, cfg=cfg, unified=unified, artifact_dir=artifact_dir, out_dir=analysis_dir, donor_ids=donor_ids, bundle=bundle, condition_encoder=condition_encoder, model=model, method_obj=method_obj, final_label_map=final_label_map, pca=pca, knn=knn, feature_names=feature_names)
                summary["route_a"] = route_res
                _mark_task_state(out_root, task_state, "route_a", _task_ok_payload(route_res if isinstance(route_res, dict) else {"result": str(route_res)}))
            except Exception as exc:
                summary["route_a"] = f"failed: {exc}"
                _mark_task_state(out_root, task_state, "route_a", _task_fail_payload(exc))
        else:
            summary["route_a"] = {"status": "reused_checkpoint"}

    if cfg.run_mask_curve:
        if _should_run_task(cfg, task_state, "mask_curve", _mask_outputs_ok):
            try:
                if condition_encoder is None or model is None:
                    condition_encoder = ClipConditionEncoder(cfg.clip_cfg, cfg.clip_ckpt, device, condition_mode=cfg.condition_mode)
                    model, method_obj = _build_model_and_method(unified, condition_encoder, device)
                importance_path = _find_importance_file(analysis_dir)
                mask_res = _run_mask_curve(legacy=legacy, cfg=cfg, unified=unified, out_dir=analysis_dir, artifact_dir=artifact_dir, bundle=bundle, condition_encoder=condition_encoder, model=model, method_obj=method_obj, donor_ids=donor_ids, X_ref=X_ref, current_token_labels=ehr_feature_names, importance_path=importance_path)
                summary["mask_curve"] = mask_res
                if isinstance(mask_res, dict) and mask_res.get("status") == "ok":
                    _mark_task_state(out_root, task_state, "mask_curve", _task_ok_payload(mask_res))
                else:
                    _mark_task_state(out_root, task_state, "mask_curve", {"status": str(mask_res.get("status", "skipped")) if isinstance(mask_res, dict) else "skipped"})
            except Exception as exc:
                summary["mask_curve"] = f"failed: {exc}"
                _mark_task_state(out_root, task_state, "mask_curve", _task_fail_payload(exc))
        else:
            summary["mask_curve"] = {"status": "reused_checkpoint"}

    if cfg.run_hist_eval and cfg.export_npy_final:
        if _should_run_task(cfg, task_state, "hist_eval", _hist_outputs_ok):
            try:
                metrics_dir = analysis_dir / "metrics"
                plots_dir = analysis_dir / "plots"
                compare_generated_to_original(cfg.sc_csv, str(npy_dir), str(metrics_dir), str(plots_dir), cfg.feature_dims)
                hist_payload = {"metrics_dir": str(metrics_dir), "plots_dir": str(plots_dir)}
                try:
                    plot_feature_histogram_by_label(str(npy_dir), cfg.sc_csv, cfg.label_xlsx, donor_ids, str(analysis_dir / "feat_by_label"), cfg.feature_dims)
                    hist_payload["feat_by_label"] = str(analysis_dir / "feat_by_label")
                except Exception as exc:
                    summary["feat_by_label"] = f"failed: {exc}"
                _mark_task_state(out_root, task_state, "hist_eval", _task_ok_payload(hist_payload))
            except Exception as exc:
                _mark_task_state(out_root, task_state, "hist_eval", _task_fail_payload(exc))
                raise
        else:
            summary["hist_eval"] = {"status": "reused_checkpoint"}

    _save_json(out_root / "analysis_summary.json", summary)
    return summary
