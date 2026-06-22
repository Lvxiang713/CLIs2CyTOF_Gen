from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
try:
    from torchdiffeq import odeint
except Exception:
    odeint = None

from singlecell_generative_unified.models.conditions.clip_condition import ClipConditionEncoder
from singlecell_generative_unified.models.methods.ddpm import DDPMMethod


def evenly_spaced_indices(total_updates: int, intervals: int = 10) -> list[int]:
    idx = np.linspace(0, total_updates, intervals + 1)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, total_updates)
    return sorted(set(idx.tolist()))


def _artifact_stats_from_traj(traj: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    arr = traj.float().reshape(traj.shape[0], -1)
    mean_u = arr.mean(dim=1).cpu().numpy().astype(np.float32)
    std_u = arr.std(dim=1, unbiased=False).cpu().numpy().astype(np.float32)
    return mean_u, std_u


def _nan_cross_attn(num_steps: int, cell_num: int, dims: int, kv_len: int) -> torch.Tensor:
    out = torch.empty((num_steps, cell_num, dims, kv_len), dtype=torch.float32)
    out.fill_(float("nan"))
    return out


@torch.no_grad()
def make_cond_from_donor(
    donor_id: str,
    *,
    dataset,
    condition_encoder: ClipConditionEncoder,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ehr = torch.as_tensor(dataset.ehr_dict[donor_id], dtype=torch.float32, device=device).unsqueeze(0)
    return condition_encoder.encode_batch(ehr)


@torch.no_grad()
def make_cond_from_ehr_array(
    ehr_vec: np.ndarray,
    *,
    condition_encoder: ClipConditionEncoder,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ehr = torch.as_tensor(ehr_vec, dtype=torch.float32, device=device).view(1, -1)
    return condition_encoder.encode_batch(ehr)


@torch.no_grad()
def sample_flowmatching_euler_artifact(
    *,
    model: torch.nn.Module,
    cond: tuple[torch.Tensor, torch.Tensor | None],
    cell_num: int,
    dims: int,
    steps: int,
    record_cross_attn: bool,
    attn_aggregate: str = "mean",
) -> dict:
    device = next(model.parameters()).device
    x = torch.randn((1, cell_num, dims), device=device)
    total_updates = int(steps)
    snap_updates = evenly_spaced_indices(total_updates, intervals=10)
    t_vals = torch.linspace(1.0, 0.0, steps, device=device)
    dt = float((t_vals[1] - t_vals[0]).detach().cpu()) if steps > 1 else -1.0

    traj_list = []
    t_used = []
    cross_attn = None
    kv_len = int(cond[0].shape[1])

    if record_cross_attn and hasattr(model, "set_cross_attn_recording"):
        cross_attn = _nan_cross_attn(len(snap_updates), cell_num, dims, kv_len)
        model.set_cross_attn_recording(True, clear=True)

    snap_ptr = 0
    if 0 in snap_updates:
        traj_list.append(x[0].detach().cpu().float())
        t_used.append(np.float32(1.0))
        snap_ptr = 1

    updates_done = 0
    for t in t_vals:
        t_batch = t.expand(1)
        v = model(x, t_batch, cond)
        attn = None
        if record_cross_attn and hasattr(model, "collect_cross_attn"):
            attn = model.collect_cross_attn(aggregate=attn_aggregate, clear=True)
        x = x + v * (t_vals[1] - t_vals[0] if steps > 1 else 0.0)
        updates_done += 1
        if updates_done in snap_updates:
            traj_list.append(x[0].detach().cpu().float())
            t_used.append(np.float32(float(t.detach().cpu())))
            if cross_attn is not None and attn is not None:
                cross_attn[snap_ptr] = attn.detach().cpu().float()
            snap_ptr += 1

    if cross_attn is not None:
        model.set_cross_attn_recording(False, clear=False)

    traj = torch.stack(traj_list, dim=0)
    mean_u, std_u = _artifact_stats_from_traj(traj)
    return {
        "traj": traj,
        "t_used": np.asarray(t_used, dtype=np.float32),
        "dt": float(dt),
        "mean_u": mean_u,
        "std_u": std_u,
        "cross_attn": cross_attn,
        "sampled_update_indices": np.asarray(snap_updates[: len(traj_list)], dtype=np.int32),
        "analysis_mode": "multistep",
        "initial_noise_kind": "data_x0",
    }


@torch.no_grad()
def sample_flowmatching_final_from_given_x0(
    *,
    model: torch.nn.Module,
    cond: tuple[torch.Tensor, torch.Tensor | None],
    x0: np.ndarray | torch.Tensor,
    steps: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    x = torch.as_tensor(x0, dtype=torch.float32, device=device)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    t_vals = torch.linspace(1.0, 0.0, steps, device=device)
    if t_vals.numel() < 2:
        return x[0].detach().cpu().numpy().astype(np.float32)
    dt = t_vals[1] - t_vals[0]
    for t in t_vals:
        t_batch = t.expand(x.size(0))
        v = model(x, t_batch, cond)
        x = x + v * dt
    return x[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def sample_flowmatching_odeint_artifact(
    *,
    model: torch.nn.Module,
    cond: tuple[torch.Tensor, torch.Tensor | None],
    cell_num: int,
    dims: int,
    intervals: int = 10,
    record_cross_attn: bool = False,
    attn_aggregate: str = "mean",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    solver: str = "dopri5",
) -> dict:
    if odeint is None:
        raise ImportError("torchdiffeq is required for flow_method=odeint but is not installed.")
    device = next(model.parameters()).device
    x0 = torch.randn((1, cell_num, dims), device=device)
    t_span = torch.linspace(1.0, 0.0, steps=intervals + 1, device=device)

    def ode_func(t_scalar: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(1, cell_num, dims)
        t_batch = torch.full((1,), float(t_scalar), device=device)
        v = model(x, t_batch, cond)
        return v.view(-1, dims)

    sol = odeint(ode_func, x0.view(-1, dims), t_span, rtol=rtol, atol=atol, method=solver)
    traj = sol.view(intervals + 1, 1, cell_num, dims)[:, 0].detach().cpu().float()

    cross_attn = None
    kv_len = int(cond[0].shape[1])
    if record_cross_attn and hasattr(model, "set_cross_attn_recording"):
        cross_attn = _nan_cross_attn(intervals + 1, cell_num, dims, kv_len)
        model.set_cross_attn_recording(True, clear=True)
        for i, t in enumerate(t_span[1:], start=1):
            x_state = traj[i].unsqueeze(0).to(device)
            t_batch = torch.full((1,), float(t), device=device)
            _ = model(x_state, t_batch, cond)
            attn = model.collect_cross_attn(aggregate=attn_aggregate, clear=True)
            if attn is not None:
                cross_attn[i] = attn.detach().cpu().float()
        model.set_cross_attn_recording(False, clear=False)

    mean_u, std_u = _artifact_stats_from_traj(traj)
    return {
        "traj": traj,
        "t_used": t_span.detach().cpu().numpy().astype(np.float32),
        "dt": float((t_span[1] - t_span[0]).detach().cpu()),
        "mean_u": mean_u,
        "std_u": std_u,
        "cross_attn": cross_attn,
        "sampled_update_indices": np.arange(intervals + 1, dtype=np.int32),
        "analysis_mode": "multistep",
        "initial_noise_kind": "data_x0",
    }


@torch.no_grad()
def sample_ddpm_artifact(
    *,
    model: torch.nn.Module,
    method: DDPMMethod,
    cond: tuple[torch.Tensor, torch.Tensor | None],
    cell_num: int,
    dims: int,
    sampler: str,
    ddim_steps: int,
    eta: float,
    clip_denoised: bool,
    record_cross_attn: bool,
    attn_aggregate: str = "mean",
) -> dict:
    device = next(model.parameters()).device
    if sampler == "ddim":
        total_updates = int(ddim_steps)
        time_pairs = torch.linspace(method.timesteps - 1, 0, steps=ddim_steps, device=device).long()
        prev_times = torch.cat([time_pairs[1:], torch.zeros(1, device=device, dtype=torch.long)])
        schedule = list(zip(time_pairs.tolist(), prev_times.tolist()))
    else:
        total_updates = int(method.timesteps)
        schedule = [(t, max(t - 1, 0)) for t in reversed(range(method.timesteps))]

    snap_updates = evenly_spaced_indices(total_updates, intervals=10)
    x = torch.randn((1, cell_num, dims), device=device)
    traj_list = [x[0].detach().cpu().float()]
    t_used = [np.float32(1.0)]
    snap_ptr = 1
    kv_len = int(cond[0].shape[1])
    cross_attn = None
    if record_cross_attn and hasattr(model, "set_cross_attn_recording"):
        cross_attn = _nan_cross_attn(len(snap_updates), cell_num, dims, kv_len)
        model.set_cross_attn_recording(True, clear=True)

    updates_done = 0
    for t_now, t_prev in schedule:
        t = torch.full((1,), int(t_now), device=device, dtype=torch.long)
        if sampler == "ddim":
            tp = torch.full((1,), int(t_prev), device=device, dtype=torch.long)
            x = method.ddim_step(model, x, t, tp, cond, eta=eta, clip_denoised=clip_denoised)
        else:
            x = method.p_sample(model, x, t, cond, clip_denoised=clip_denoised)
        attn = None
        if record_cross_attn and hasattr(model, "collect_cross_attn"):
            attn = model.collect_cross_attn(aggregate=attn_aggregate, clear=True)
        updates_done += 1
        if updates_done in snap_updates:
            traj_list.append(x[0].detach().cpu().float())
            frac = 1.0 - updates_done / max(total_updates, 1)
            t_used.append(np.float32(frac))
            if cross_attn is not None and attn is not None:
                cross_attn[snap_ptr] = attn.detach().cpu().float()
            snap_ptr += 1

    if cross_attn is not None:
        model.set_cross_attn_recording(False, clear=False)

    traj = torch.stack(traj_list, dim=0)
    mean_u, std_u = _artifact_stats_from_traj(traj)
    return {
        "traj": traj,
        "t_used": np.asarray(t_used, dtype=np.float32),
        "dt": float(-1.0 / max(total_updates, 1)),
        "mean_u": mean_u,
        "std_u": std_u,
        "cross_attn": cross_attn,
        "sampled_update_indices": np.asarray(snap_updates[: len(traj_list)], dtype=np.int32),
        "analysis_mode": "multistep",
        "initial_noise_kind": "data_x0",
    }


@torch.no_grad()
def sample_ddpm_final_from_given_noise(
    *,
    model: torch.nn.Module,
    method: DDPMMethod,
    cond: tuple[torch.Tensor, torch.Tensor | None],
    x_init: np.ndarray | torch.Tensor,
    sampler: str = "ddpm",
    ddim_steps: int = 50,
    eta: float = 0.0,
    clip_denoised: bool = True,
) -> np.ndarray:
    device = next(model.parameters()).device
    x = torch.as_tensor(x_init, dtype=torch.float32, device=device)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    batch_size = x.size(0)
    sampler = str(sampler).lower()
    if sampler == "ddim":
        time_pairs = torch.linspace(method.timesteps - 1, 0, steps=ddim_steps, device=device).long()
        prev_times = torch.cat([time_pairs[1:], torch.zeros(1, device=device, dtype=torch.long)])
        for t_now, t_prev in zip(time_pairs, prev_times):
            t = torch.full((batch_size,), int(t_now.item()), device=device, dtype=torch.long)
            tp = torch.full((batch_size,), int(t_prev.item()), device=device, dtype=torch.long)
            x = method.ddim_step(model, x, t, tp, cond, eta=eta, clip_denoised=clip_denoised)
    else:
        for i in reversed(range(0, method.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = method.p_sample(model, x, t, cond, clip_denoised=clip_denoised)
    return x[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def sample_one_step_artifact(
    *,
    model: torch.nn.Module,
    cond: tuple[torch.Tensor, torch.Tensor | None],
    cell_num: int,
    dims: int,
    latent_dim: int,
    mode: str,
    record_cross_attn: bool,
    attn_aggregate: str = "mean",
) -> dict:
    device = next(model.parameters()).device
    z0 = torch.randn((1, cell_num, latent_dim), device=device)
    cross_attn = None
    kv_len = int(cond[0].shape[1])

    if record_cross_attn and hasattr(model, "set_cross_attn_recording"):
        model.set_cross_attn_recording(True, clear=True)

    x_final = model(z0, cond)

    if record_cross_attn and hasattr(model, "collect_cross_attn"):
        attn = model.collect_cross_attn(aggregate=attn_aggregate, clear=True)
        if attn is not None:
            cross_attn = _nan_cross_attn(1, cell_num, dims, kv_len)
            cross_attn[0] = attn.detach().cpu().float()
        model.set_cross_attn_recording(False, clear=False)

    traj = x_final.detach().cpu().float()
    traj = traj if traj.dim() == 3 else traj.unsqueeze(0)
    traj = traj[0].unsqueeze(0)
    mean_u, std_u = _artifact_stats_from_traj(traj)
    return {
        "traj": traj,
        "t_used": np.asarray([0.0], dtype=np.float32),
        "dt": 0.0,
        "mean_u": mean_u,
        "std_u": std_u,
        "cross_attn": cross_attn,
        "sampled_update_indices": np.asarray([0], dtype=np.int32),
        "analysis_mode": "one_step",
        "initial_noise_kind": f"latent_{mode}",
        "latent_noise": z0[0].detach().cpu().float(),
    }


@torch.no_grad()
def sample_one_step_final_from_given_latent(
    *,
    model: torch.nn.Module,
    cond: tuple[torch.Tensor, torch.Tensor | None],
    z_init: np.ndarray | torch.Tensor,
) -> np.ndarray:
    device = next(model.parameters()).device
    z = torch.as_tensor(z_init, dtype=torch.float32, device=device)
    if z.dim() == 2:
        z = z.unsqueeze(0)
    x = model(z, cond)
    return x[0].detach().cpu().numpy().astype(np.float32)


def save_donor_artifact(output_dir: str | os.PathLike[str], donor_id: str, artifact: dict) -> str:
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{donor_id}.pt")
    torch.save({"donor_id": donor_id, **artifact}, path)
    return path


def list_artifact_paths(output_dir: str | os.PathLike[str]) -> list[str]:
    output_dir = Path(output_dir)
    return sorted(str(p) for p in output_dir.glob("*.pt"))
