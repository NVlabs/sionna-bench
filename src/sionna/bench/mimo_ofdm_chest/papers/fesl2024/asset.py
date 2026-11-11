# SPDX-FileCopyrightText: Copyright (c) 2026 Benedikt Fesl
# SPDX-License-Identifier: Apache-2.0
"""Dataset generation, training, and model assets for Fesl et al. 2024."""


from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.object import Object
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin

from sionna.bench.mimo_ofdm_chest.asset_store import (
    build_asset_manifest,
    external_asset_dir,
    load_asset_manifest,
    paper_asset_dir,
)
from sionna.bench.mimo_ofdm_chest.benchmarking import (
    build_channel_source,
    set_random_seed,
)
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig
from sionna.bench.mimo_ofdm_chest.persistence import (
    sha256_file,
    sha256_fingerprint,
    write_json_atomic,
    write_npz_atomic,
)

from .estimator import _DiffusionSchedule, _NoisePredictionCNN

__all__ = ["generate_training_data", "load_training_data", "train_model", "load_model"]

_DATA_TYPE = "diffusion-training-data"
_MODEL_TYPE = "diffusion-prior"


class _PilotGroups(Object, _PilotGroupMixin):
    def __init__(self, resource_grid: ResourceGrid, cfg: SystemConfig) -> None:
        super().__init__(precision=cfg.precision, device=cfg.device)
        self._init_pilot_groups(resource_grid.pilot_pattern, group_over=())


def _tensor_identity(value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().cpu().contiguous().numpy()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _compatibility(resource_grid: ResourceGrid, cfg: SystemConfig) -> dict[str, Any]:
    if cfg.direction != "uplink" or cfg.bs_polarization != "single":
        raise ValueError(
            "Diffusion assets require an uplink single-polarized receive array."
        )
    if cfg.ue_polarization != "single" or cfg.num_streams_per_tx != 1:
        raise ValueError(
            "Diffusion data generation requires single-antenna, single-stream UEs."
        )
    groups = _PilotGroups(resource_grid, cfg)
    values = groups._gather_pilot_groups(resource_grid.pilot_pattern.pilots)
    snapshot = cfg.config_snapshot()
    return {
        "model_version": "fesl2024-cnn-v1",
        "architecture": {
            "pre": [2, 32, 64],
            "post": [64, 43, 22, 2],
            "embedding": 16,
            "kernel": [3, 3],
            "padding": "zeros-same",
        },
        "system": snapshot["system"],
        "channel_source": snapshot["channel_source"],
        "source_precision": cfg.precision,
        "receive_array_shape": [cfg.bs_num_rows, cfg.bs_num_cols],
        "antenna_order": "row-plus-rows-times-column",
        "angular_transform": "negative-sign-fft2-ortho-physical-rows-cols",
        "representation": "real-imaginary-convolution-channels-complex-noise-variance-one",
        "normalization": "pooled-training-only-mean-complex-entry-power",
        "pooling": "all-active-pilot-receive-vectors-equal-weight",
        "timestep_convention": "coefficient-index-t-is-paper-step-t-plus-one-inclusive",
        "schedule": {
            "length": 100,
            "beta_sha256": _tensor_identity(
                _DiffusionSchedule(dtype=torch.float64).betas
            )["sha256"],
        },
        "pilot_groups": {
            "coordinates": _tensor_identity(groups._pilot_group_coords),
            "values": _tensor_identity(values),
            "powers": _tensor_identity(values.abs().square()),
        },
    }


def _asset_dir(
    cfg: SystemConfig,
    asset_type: str,
    artifact_id: str,
    asset_root: Optional[str | Path],
) -> Path:
    if asset_root is None:
        return paper_asset_dir(cfg, asset_type, artifact_id, __file__)
    return external_asset_dir(
        cfg, "fesl2024", asset_type, artifact_id, asset_root=asset_root
    )


def _temporary_destination(target: Path, overwrite: bool) -> Path:
    if target.exists() and not overwrite:
        raise FileExistsError(f"Asset destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))


def _publish(temporary: Path, target: Path, overwrite: bool) -> None:
    # A complete sibling directory is promoted, never partially filled in place.
    if target.exists():
        if not overwrite:
            raise FileExistsError(target)
        backup = target.with_name(f".{target.name}-replaced-{time.time_ns()}")
        os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except BaseException:
            os.replace(backup, target)
            raise
        shutil.rmtree(backup)
    else:
        os.replace(temporary, target)


def _seed(role: str, base: int, epoch: int = 0) -> int:
    """Stable local-generator key; never Python's process-randomized hash()."""
    key = json.dumps(
        ["fesl2024-rng-v1", role, int(base), int(epoch)], separators=(",", ":")
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (
        2**63 - 1
    )


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "source_files_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("estimator.py", "asset.py", "tuning.py", "run_benchmark.py")
            if Path(__file__).with_name(name).is_file()
        },
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "sionna": version("sionna"),
        "torch_threads": torch.get_num_threads(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def generate_training_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    asset_root: str | Path,
    artifact_id: str = "training-data",
    num_train_channels: int = 6400,
    num_validation_channels: int = 256,
    generation_batch_size: int = 16,
    train_seed: int = 24250000,
    validation_seed: int = 24260000,
    purpose: str = "canonical",
    overwrite: bool = False,
) -> Path:
    r"""Generate and persist a deterministic bank of clean receive-array channels.

    The asset contains the clean channel vector at every active pilot resource
    element of independent system-channel realizations, together with the
    training-population channel power used for normalization. Training and
    validation realizations use disjoint deterministic propagation seeds. The
    completed asset is strictly validated before its manifest path is returned.

    :param resource_grid: Sionna resource grid whose pilot geometry selects the
        receive-array vectors retained from each channel realization.
    :param cfg: System and channel configuration used to generate the channel
        realizations.
    :param asset_root: Writable external root beneath which the generated payloads
        and manifest are stored.
    :param artifact_id: Identifier of the generated dataset asset.
    :param num_train_channels: Number of complete system-channel realizations in
        the training split.
    :param num_validation_channels: Number of complete system-channel
        realizations in the validation split.
    :param generation_batch_size: Number of realizations generated per batch.
        Batch ``i`` of a split uses the propagation seed ``split_seed + i``.
    :param train_seed: Base propagation seed of the training split.
    :param validation_seed: Base propagation seed of the validation split. The
        seed ranges of the two splits must not overlap.
    :param purpose: Label recorded in the manifest that distinguishes canonical
        data from smoke and development data. Loaders refuse data whose purpose
        does not match the requested one.
    :param overwrite: If `True`, replace an existing asset with the same identity.
        If `False`, raise an error when that asset already exists.

    :returns: Path to the strictly validated ``manifest.json`` of the generated
        training-data asset.
    """
    for name, count in (
        ("train", num_train_channels),
        ("validation", num_validation_channels),
        ("generation_batch_size", generation_batch_size),
    ):
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"{name} count must be a positive integer.")
    train_batches = math.ceil(num_train_channels / generation_batch_size)
    val_batches = math.ceil(num_validation_channels / generation_batch_size)
    if max(train_seed, validation_seed) <= min(
        train_seed + train_batches - 1, validation_seed + val_batches - 1
    ):
        raise ValueError("Training and validation propagation seed ranges overlap.")
    compatibility = _compatibility(resource_grid, cfg)
    target = _asset_dir(cfg, _DATA_TYPE, artifact_id, asset_root)
    temporary = _temporary_destination(target, overwrite)
    started = time.perf_counter()
    groups = _PilotGroups(resource_grid, cfg)
    coords = groups._pilot_group_coords[:, 0]
    num_groups = int(coords.shape[0])
    recipe = {
        "num_train_channels": num_train_channels,
        "num_validation_channels": num_validation_channels,
        "generation_batch_size": generation_batch_size,
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "purpose": purpose,
        "seed_addressing": "split-base-plus-batch-index",
    }
    training_power_sum = 0.0
    training_sum = 0j
    try:
        for split, count, base in (
            ("train", num_train_channels, train_seed),
            ("validation", num_validation_channels, validation_seed),
        ):
            array = np.lib.format.open_memmap(
                temporary / f"{split}.npy",
                mode="w+",
                dtype=np.complex64,
                shape=(count, num_groups, cfg.num_rx_ant),
            )
            # Match the unnormalized shared channel-source path, without bit/noise work.
            set_random_seed(base)
            source = build_channel_source(cfg, resource_grid)
            generator = GenerateOFDMChannel(
                source.channel_model,
                resource_grid,
                normalize_channel=False,
                precision=cfg.precision,
                device=cfg.device,
            )
            with torch.no_grad():
                for batch, offset in enumerate(range(0, count, generation_batch_size)):
                    size = min(generation_batch_size, count - offset)
                    set_random_seed(base + batch)
                    source.sample(size)
                    channels = generator(size)
                    vectors = channels[
                        :, :, :, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
                    ]
                    vectors = vectors.movedim(-2, -1).reshape(
                        size, num_groups, cfg.num_rx_ant
                    )
                    values = vectors.cpu().numpy().astype(np.complex64)
                    if not np.isfinite(values).all():
                        raise ValueError(
                            "Channel generation produced non-finite values."
                        )
                    array[offset : offset + size] = values
                    if split == "train":
                        training_power_sum += float(
                            np.square(np.abs(values).astype(np.float64)).sum()
                        )
                        training_sum += complex(values.astype(np.complex128).sum())
            array.flush()
            del array
        num_entries = num_train_channels * num_groups * cfg.num_rx_ant
        power = training_power_sum / num_entries
        if not math.isfinite(power) or power <= 0:
            raise ValueError("Training bank has invalid population channel power.")
        write_npz_atomic(
            temporary / "normalization.npz",
            channel_power=np.asarray(power, dtype=np.float64),
        )
        manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=_DATA_TYPE,
            artifact_id=artifact_id,
            compatibility=compatibility,
            generation_recipe=recipe,
            payload_paths=[
                temporary / "train.npy",
                temporary / "validation.npy",
                temporary / "normalization.npz",
            ],
            metadata={
                "groups_per_realization": num_groups,
                "derived_train_examples": num_train_channels * num_groups,
                "derived_validation_examples": num_validation_channels * num_groups,
                "training_complex_mean": [
                    training_sum.real / num_entries,
                    training_sum.imag / num_entries,
                ],
                "generation_seconds": time.perf_counter() - started,
                "runtime": _runtime(),
            },
        )
        write_json_atomic(temporary / "manifest.json", manifest)
        _load_data(temporary / "manifest.json", compatibility)
        _publish(temporary, target, overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    load_training_data(resource_grid, cfg, manifest_path=target / "manifest.json")
    return target / "manifest.json"


def _load_data(
    path: Path, compatibility: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
    manifest = load_asset_manifest(
        path,
        asset_dir=path.parent,
        expected_asset_type=_DATA_TYPE,
        expected_compatibility=compatibility,
    )
    files = {record["path"] for record in manifest["files"]}
    if files != {"train.npy", "validation.npy", "normalization.npz"}:
        raise ValueError("Diffusion dataset has unexpected payload files.")
    groups = compatibility["pilot_groups"]["coordinates"]["shape"][0]
    antennas = math.prod(compatibility["receive_array_shape"])
    arrays = {}
    training_power_sum = 0.0
    for split in ("train", "validation"):
        count = manifest["generation_recipe"].get(f"num_{split}_channels")
        array = np.load(path.parent / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(count, int)
            or count < 1
            or array.shape != (count, groups, antennas)
            or array.dtype != np.complex64
        ):
            raise ValueError(f"Invalid {split} channel bank shape/dtype/count.")
        if manifest["metadata"].get(f"derived_{split}_examples") != count * groups:
            raise ValueError(
                "Dataset derived-example accounting does not match its channels."
            )
        for offset in range(0, count, 16):
            chunk = array[offset : offset + 16]
            if not np.isfinite(chunk).all():
                raise ValueError("Dataset contains non-finite channel entries.")
            if split == "train":
                training_power_sum += float(
                    np.square(np.abs(chunk).astype(np.float64)).sum()
                )
        arrays[split] = array
    with np.load(path.parent / "normalization.npz", allow_pickle=False) as archive:
        if (
            archive.files != ["channel_power"]
            or archive["channel_power"].shape != ()
            or archive["channel_power"].dtype != np.float64
        ):
            raise ValueError("Invalid channel-power payload.")
        power = float(archive["channel_power"])
    if not math.isfinite(power) or power <= 0:
        raise ValueError("Invalid training channel power.")
    expected_power = training_power_sum / arrays["train"].size
    if not math.isclose(power, expected_power, rel_tol=1e-12):
        raise ValueError("Channel power does not match the training-only statistic.")
    return manifest, arrays, power


def load_training_data(
    resource_grid: ResourceGrid, cfg: SystemConfig, *, manifest_path: str | Path
) -> tuple[dict[str, np.ndarray], float, dict[str, Any]]:
    """Strictly validate a clean bank and return memory maps, power and manifest."""
    manifest, arrays, power = _load_data(
        Path(manifest_path), _compatibility(resource_grid, cfg)
    )
    if (
        manifest["scenario_id"] != cfg.scenario_id
        or Path(manifest_path).parent.name != manifest["artifact_id"]
    ):
        raise ValueError("Dataset identity does not match scenario or directory.")
    return arrays, power, manifest


def _angular_batch(
    values: np.ndarray, shape: tuple[int, int], power: float, device: str
) -> torch.Tensor:
    rows, cols = shape
    # Keep mmap access and complex transforms on CPU, independently of Sionna's
    # device. Only the small real angular minibatch enters native PyTorch MPS.
    tensor = torch.as_tensor(np.array(values, copy=True), device="cpu")
    array = (tensor / math.sqrt(power)).reshape(-1, cols, rows).transpose(-1, -2)
    angular = torch.fft.fft2(array, norm="ortho")
    return torch.stack((angular.real, angular.imag), dim=1).to(device)


def _cpu_tree(value: Any) -> Any:
    """Serialize optimizer tensors on CPU without changing their dtypes."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _save_training_checkpoint(directory: Path, payload: dict[str, Any]) -> None:
    """Commit an immutable epoch first, then atomically advance its hash pointer."""
    path = directory / f"epoch-{payload['completed_epoch']:04d}.pt"
    descriptor, name = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(_cpu_tree(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        pointer = {
            "file": path.name,
            "sha256": sha256_file(path),
            "completed_epoch": payload["completed_epoch"],
        }
        write_json_atomic(path.with_suffix(".json"), pointer)
        write_json_atomic(directory / "latest.json", pointer)
        if (
            payload["best_epoch"] == payload["completed_epoch"]
            and payload["completed_epoch"]
        ):
            write_json_atomic(directory / "best.json", pointer)
        write_json_atomic(directory / "history.json", payload["history"])
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_training_checkpoint(
    directory: Path, context: dict[str, Any]
) -> dict[str, Any]:
    pointer = json.loads((directory / "latest.json").read_text())
    name = pointer["file"]
    if Path(name).name != name:
        raise ValueError("Invalid checkpoint path.")
    path = directory / name
    if sha256_file(path) != pointer["sha256"]:
        raise ValueError("Training checkpoint checksum mismatch.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_version") != 1 or payload.get("context") != context:
        raise ValueError(
            "Resume context changed (data, recipe, source, runtime or device)."
        )
    epoch = payload["completed_epoch"]
    if epoch != pointer["completed_epoch"] or epoch != len(payload["history"]):
        raise ValueError("Checkpoint epoch/history mismatch.")
    if [row["epoch"] for row in payload["history"]] != list(range(1, epoch + 1)):
        raise ValueError("Checkpoint history is not contiguous.")
    if not 0 <= payload["best_epoch"] <= epoch or payload["stale_epochs"] < 0:
        raise ValueError("Invalid checkpoint selection/stopping state.")
    return payload


def _draws(
    count: int, shape: tuple[int, int], seed: int, epoch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw in stable example-ID order before optimizer batching/shuffling.

    A sample's random address is (role/seed/epoch key, example ID, component
    offset) in the CPU generator stream. Batch size cannot change that address.
    The full noise tensor costs 128 bytes/example on a 4x4 array.
    """
    noise_rng = torch.Generator().manual_seed(_seed("diffusion-noise", seed, epoch))
    time_rng = torch.Generator().manual_seed(_seed("diffusion-timestep", seed, epoch))
    noise = torch.randn((count, 2, *shape), generator=noise_rng) / math.sqrt(2.0)
    timesteps = torch.randint(100, (count,), generator=time_rng)
    return noise, timesteps


def _batches(order: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]:
    return iter(order.split(batch_size))


def _train_state(
    arrays: dict[str, np.ndarray],
    power: float,
    shape: tuple[int, int],
    *,
    device: str,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    patience_epochs: int,
    relative_min_delta: float,
    initialization_seed: int,
    noise_seed: int,
    validation_noise_seed: int,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_context: Optional[dict[str, Any]] = None,
    resume: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if (
        epochs < 1
        or optimizer_batch_size < 1
        or learning_rate <= 0
        or not math.isfinite(learning_rate)
    ):
        raise ValueError(
            "Training epochs, batch size and learning rate must be positive."
        )
    if patience_epochs < 0 or not 0 <= relative_min_delta < 1:
        raise ValueError("Invalid early-stopping settings.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        model = _NoisePredictionCNN().to(device)
    schedule = _DiffusionSchedule(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train = arrays["train"].reshape(-1, math.prod(shape))
    validation = arrays["validation"].reshape(-1, math.prod(shape))
    validation_noise, validation_times = _draws(
        len(validation), shape, validation_noise_seed, 0
    )
    best_loss = math.inf
    stopping_loss = math.inf
    best_state: dict[str, torch.Tensor] = {}
    stale = 0
    history = []
    best_epoch = 0
    completed_epoch = 0
    previous_seconds = 0.0
    if resume and checkpoint_dir is None:
        raise ValueError("Resume requires an explicit external checkpoint directory.")
    if checkpoint_dir is not None:
        if checkpoint_context is None:
            raise ValueError("Checkpointing requires a frozen context.")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not resume and (checkpoint_dir / "latest.json").exists():
            raise FileExistsError(
                "Training already has checkpoints; explicitly resume it."
            )
        if resume:
            saved = _load_training_checkpoint(checkpoint_dir, checkpoint_context)
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            # Adam's scalar step remains CPU for non-capturable optimizers;
            # load_state_dict places moment tensors on the parameter device.
            completed_epoch, history = saved["completed_epoch"], saved["history"]
            best_state, best_loss, best_epoch = (
                saved["best_state"],
                saved["best_loss"],
                saved["best_epoch"],
            )
            stopping_loss, stale = saved["stopping_loss"], saved["stale_epochs"]
            previous_seconds = saved["training_seconds"]
    started = time.perf_counter()

    def checkpoint() -> None:
        if checkpoint_dir is not None:
            _save_training_checkpoint(
                checkpoint_dir,
                {
                    "checkpoint_version": 1,
                    "context": checkpoint_context,
                    "completed_epoch": completed_epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_state": best_state,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "stopping_loss": stopping_loss,
                    "stale_epochs": stale,
                    "history": history,
                    "training_seconds": previous_seconds
                    + time.perf_counter()
                    - started,
                },
            )

    def progress(phase: str, epoch: int, batch: int = 0) -> None:
        if checkpoint_dir is not None:
            write_json_atomic(
                checkpoint_dir / "progress.json",
                {
                    "pid": os.getpid(),
                    "phase": phase,
                    "epoch": epoch,
                    "completed_epoch": completed_epoch,
                    "batch": batch,
                    "batches_per_epoch": math.ceil(len(train) / optimizer_batch_size),
                    "training_device": device,
                    "unix_time": time.time(),
                },
            )
            if (checkpoint_dir / "STOP_REQUEST").exists():
                raise InterruptedError(
                    "Stop requested; resume replays the incomplete epoch."
                )

    if checkpoint_dir is not None and not resume:
        checkpoint()  # Epoch zero is a valid recovery point before the first update.
    progress("ready", completed_epoch)
    for epoch in range(completed_epoch, epochs):
        if patience_epochs and stale >= patience_epochs:
            break
        epoch_started = time.perf_counter()
        noise, timesteps = _draws(len(train), shape, noise_seed, epoch)
        shuffle_rng = torch.Generator().manual_seed(_seed("shuffle", noise_seed, epoch))
        order = torch.randperm(len(train), generator=shuffle_rng)
        train_loss = 0.0
        model.train()
        for batch, indices in enumerate(_batches(order, optimizer_batch_size), 1):
            clean = _angular_batch(train[indices.numpy()], shape, power, device)
            epsilon = noise[indices].to(device)
            steps = timesteps[indices].to(device)
            predicted = model(schedule.forward_noise(clean, epsilon, steps), steps)
            loss = torch.nn.functional.mse_loss(predicted, epsilon)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite diffusion training loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(indices)
            if batch == 1 or batch % 100 == 0:
                progress("training", epoch + 1, batch)
        model.eval()
        validation_loss = 0.0
        progress("validation", epoch + 1)
        with torch.no_grad():
            for batch, indices in enumerate(
                _batches(torch.arange(len(validation)), optimizer_batch_size), 1
            ):
                clean = _angular_batch(
                    validation[indices.numpy()], shape, power, device
                )
                epsilon = validation_noise[indices].to(device)
                steps = validation_times[indices].to(device)
                loss = torch.nn.functional.mse_loss(
                    model(schedule.forward_noise(clean, epsilon, steps), steps), epsilon
                )
                validation_loss += float(loss) * len(indices)
                if batch % 100 == 0:
                    progress("validation", epoch + 1, batch)
        validation_loss /= len(validation)
        if not math.isfinite(validation_loss):
            raise FloatingPointError("Non-finite diffusion validation loss.")
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": train_loss / len(train),
                "validation_loss": validation_loss,
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if validation_loss < stopping_loss * (1.0 - relative_min_delta):
            stopping_loss, stale = validation_loss, 0
        else:
            stale += 1
        completed_epoch = epoch + 1
        checkpoint()
        if checkpoint_dir is not None:
            print(json.dumps(history[-1]), flush=True)
        progress("epoch_complete", completed_epoch)
        del noise, timesteps, order
        if patience_epochs and stale >= patience_epochs:
            break
    return best_state, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "training_seconds": previous_seconds + time.perf_counter() - started,
        "stale_epochs": stale,
        "completed_epochs": completed_epoch,
        "stopped_early": completed_epoch < epochs,
        "training_device": device,
        "runtime": _runtime(),
    }


def train_model(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    training_data_manifest: str | Path,
    asset_root: str | Path,
    artifact_id: str = "canonical",
    epochs: int = 200,
    optimizer_batch_size: int = 512,
    learning_rate: float = 1e-4,
    patience_epochs: int = 20,
    relative_min_delta: float = 1e-4,
    initialization_seed: int = 24270000,
    noise_seed: int = 24280000,
    validation_noise_seed: int = 24290000,
    overwrite: bool = False,
    training_device: Optional[str] = None,
    checkpoint_dir: Optional[str | Path] = None,
    resume: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, Path]:
    r"""Train and persist the noise-prediction CNN of the diffusion prior.

    One model serves all diffusion timesteps and SNRs. The supplied training-data
    asset provides fixed clean channel realizations; the diffusion timesteps and
    noise of every epoch are drawn from deterministic seeded streams. Training
    stops early when the validation loss no longer improves, the state with the
    lowest validation loss is restored, and the completed asset is persisted
    atomically and strictly reloaded before being returned.

    .. rubric:: Deviations from the Paper

    - The released recipe trains on 100,000 training and 10,000 validation
      channel matrices for up to 500 epochs. This implementation uses the
      benchmark's deterministic bank of complete system-channel realizations,
      by default 6,400 for training and 256 for validation, each contributing
      one receive-array vector per active pilot resource element, together with a
      maximum epoch count and validation-loss early stopping.
    - The paper's unit-average-power convention is implemented by dividing the
      clean channels by the square root of the training-population channel power
      recorded in the dataset asset.

    :param resource_grid: Resource grid defining the pilot geometry of the dataset.
    :param cfg: System and channel configuration defining the receive-array
        shape, numerical precision, and device.
    :param training_data_manifest: Manifest of the fixed training-data asset.
    :param asset_root: Writable external root beneath which the model payload and
        manifest are stored.
    :param artifact_id: Identifier of the generated model asset.
    :param epochs: Maximum number of complete training epochs.
    :param optimizer_batch_size: Number of receive-array examples in each
        optimizer update.
    :param learning_rate: Adam learning rate.
    :param patience_epochs: Stop after this many epochs without the required
        validation-loss improvement. Zero disables early stopping.
    :param relative_min_delta: Minimum relative validation-loss reduction counted
        as an improvement.
    :param initialization_seed: Seed of the model initialization.
    :param noise_seed: Base seed of the per-epoch diffusion timesteps, noise, and
        example order.
    :param validation_noise_seed: Seed of the fixed validation timesteps and
        noise.
    :param overwrite: If `True`, replace an existing asset with the same identity.
        If `False`, raise an error when that asset already exists.
    :param training_device: Torch device used for training. If `None`, the
        device of ``cfg`` is used.
    :param checkpoint_dir: External directory for complete-epoch checkpoints and
        progress records. If `None`, no checkpoints are written.
    :param resume: If `True`, continue training from the latest checkpoint in
        ``checkpoint_dir``.

    :returns: The trained model state, the training-population channel power as
        a tensor, and the path to the strictly validated ``manifest.json``.
    """
    selected_device = str(torch.device(training_device or cfg.device))
    if selected_device.startswith("mps"):
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS training is unavailable; no CPU fallback is allowed."
            )
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError(
                "Disable PYTORCH_ENABLE_MPS_FALLBACK before importing PyTorch."
            )
    arrays, power, dataset = load_training_data(
        resource_grid, cfg, manifest_path=training_data_manifest
    )
    target = _asset_dir(cfg, _MODEL_TYPE, artifact_id, asset_root)
    recipe = {
        "dataset_manifest_sha256": sha256_file(training_data_manifest),
        "dataset_artifact_id": dataset["artifact_id"],
        "dataset_files": dataset["files"],
        "purpose": dataset["generation_recipe"]["purpose"],
        "epochs": epochs,
        "optimizer_batch_size": optimizer_batch_size,
        "learning_rate": learning_rate,
        "patience_epochs": patience_epochs,
        "relative_min_delta": relative_min_delta,
        "initialization_seed": initialization_seed,
        "noise_seed": noise_seed,
        "validation_noise_seed": validation_noise_seed,
        "optimizer": "Adam-default-moments-zero-weight-decay-constant-lr",
        "objective": "unweighted-epsilon-mse-real-noise-variance-half",
        "rng": "sha256-role-seed-epoch-and-stable-example-stream-offset-v1",
        "training_device": selected_device,
        "preprocessing_device": "cpu",
    }
    runtime = _runtime()
    context = {
        "recipe": recipe,
        "power": power,
        "shape": [cfg.bs_num_rows, cfg.bs_num_cols],
        "runtime": runtime,
        "checkpoint_policy": "complete-epoch-replay-v1",
    }
    checkpoint_path = None if checkpoint_dir is None else Path(checkpoint_dir).resolve()
    if checkpoint_path is not None and checkpoint_path.is_relative_to(
        Path(__file__).resolve().parent
    ):
        raise ValueError("Checkpoints require an explicit external directory.")
    lock = None
    temporary = None
    try:
        if checkpoint_path is not None:
            import fcntl  # POSIX is needed only for explicit checkpoint writers.

            checkpoint_path.mkdir(parents=True, exist_ok=True)
            lock = (checkpoint_path / "writer.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume and target.exists():
            if checkpoint_path is None:
                raise ValueError(
                    "Publication recovery requires the original checkpoint directory."
                )
            saved = _load_training_checkpoint(checkpoint_path, context)
            finished = saved["completed_epoch"] == epochs or (
                patience_epochs > 0 and saved["stale_epochs"] >= patience_epochs
            )
            existing, existing_power = _load_model(
                target / "manifest.json",
                _compatibility(resource_grid, cfg),
                recipe["purpose"],
            )
            manifest = load_asset_manifest(target / "manifest.json")
            if (
                not finished
                or manifest["generation_recipe"] != recipe
                or manifest["metadata"]["runtime"] != runtime
                or manifest["metadata"]["best_epoch"] != saved["best_epoch"]
                or manifest["metadata"]["best_validation_loss"] != saved["best_loss"]
                or float(existing_power) != power
                or any(
                    not torch.equal(existing[key], saved["best_state"][key])
                    for key in existing
                )
            ):
                raise ValueError(
                    "Existing publication does not match the completed frozen fit."
                )
            return load_model(
                resource_grid,
                cfg,
                asset_root=asset_root,
                artifact_id=artifact_id,
                expected_purpose=recipe["purpose"],
            )
        temporary = _temporary_destination(target, overwrite)
        state, metadata = _train_state(
            arrays,
            power,
            (cfg.bs_num_rows, cfg.bs_num_cols),
            device=selected_device,
            epochs=epochs,
            optimizer_batch_size=optimizer_batch_size,
            learning_rate=learning_rate,
            patience_epochs=patience_epochs,
            relative_min_delta=relative_min_delta,
            initialization_seed=initialization_seed,
            noise_seed=noise_seed,
            validation_noise_seed=validation_noise_seed,
            checkpoint_dir=checkpoint_path,
            checkpoint_context=context,
            resume=resume,
        )
        payload = {
            f"state_{index:03d}": value.numpy()
            for index, value in enumerate(state.values())
        }
        payload.update(
            channel_power=np.asarray(power, dtype=np.float64),
            betas=_DiffusionSchedule(dtype=torch.float64).betas.numpy(),
        )
        write_npz_atomic(temporary / "diffusion_params.npz", **payload)
        metadata.update(
            state_keys=list(state),
            parameter_count=sum(v.numel() for v in state.values()),
        )
        manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=_MODEL_TYPE,
            artifact_id=artifact_id,
            compatibility=_compatibility(resource_grid, cfg),
            generation_recipe=recipe,
            payload_paths=[temporary / "diffusion_params.npz"],
            metadata=metadata,
        )
        write_json_atomic(temporary / "manifest.json", manifest)
        _load_model(
            temporary / "manifest.json",
            _compatibility(resource_grid, cfg),
            recipe["purpose"],
        )
        _publish(temporary, target, overwrite)
    finally:
        if lock is not None:
            lock.close()
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
    return load_model(
        resource_grid,
        cfg,
        asset_root=asset_root,
        artifact_id=artifact_id,
        expected_purpose=recipe["purpose"],
    )


def _load_model(
    path: Path, compatibility: dict[str, Any], purpose: str
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    manifest = load_asset_manifest(
        path,
        asset_dir=path.parent,
        expected_asset_type=_MODEL_TYPE,
        expected_compatibility=compatibility,
    )
    if manifest["generation_recipe"].get("purpose") != purpose:
        raise ValueError(
            f"Expected {purpose} inference asset; refusing a different preparation purpose."
        )
    if [f["path"] for f in manifest["files"]] != ["diffusion_params.npz"]:
        raise ValueError("Unexpected inference payload files.")
    with torch.random.fork_rng(devices=[]):
        expected = _NoisePredictionCNN().state_dict()
    if manifest["metadata"].get("state_keys") != list(expected):
        raise ValueError("Checkpoint state keys do not match the diffusion CNN.")
    state = {}
    with np.load(path.parent / "diffusion_params.npz", allow_pickle=False) as archive:
        names = {f"state_{i:03d}" for i in range(len(expected))} | {
            "betas",
            "channel_power",
        }
        if set(archive.files) != names:
            raise ValueError("Checkpoint array names do not match its state mapping.")
        for index, (key, template) in enumerate(expected.items()):
            array = archive[f"state_{index:03d}"]
            if (
                array.dtype != np.float32
                or array.shape != tuple(template.shape)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"Invalid checkpoint tensor: {key}")
            state[key] = torch.from_numpy(array.copy())
        betas = archive["betas"]
        if betas.dtype != np.float64 or not np.array_equal(
            betas, _DiffusionSchedule(dtype=torch.float64).betas.numpy()
        ):
            raise ValueError(
                "Checkpoint does not use the canonical WCL diffusion schedule."
            )
        power = archive["channel_power"]
        if (
            power.dtype != np.float64
            or power.shape != ()
            or not np.isfinite(power)
            or power <= 0
        ):
            raise ValueError("Invalid checkpoint training-power scalar.")
        power_tensor = torch.as_tensor(power.copy())
    return state, power_tensor


def load_model(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    asset_root: Optional[str | Path] = None,
    artifact_id: str = "canonical",
    expected_purpose: str = "canonical",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, Path]:
    """Strictly load prepared tensors plus manifest path, without writes/training."""
    target = _asset_dir(cfg, _MODEL_TYPE, artifact_id, asset_root)
    path = target / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing diffusion asset: {path}. Prepare explicitly with "
            "python -m sionna.bench.mimo_ofdm_chest.papers.fesl2024.run_benchmark "
            "--prepare-assets --asset-root <path> [--smoke], then evaluate it with "
            "the same module --asset-root <path> [--smoke]."
        )
    state, power = _load_model(
        path, _compatibility(resource_grid, cfg), expected_purpose
    )
    manifest = load_asset_manifest(
        path, expected_scenario_id=cfg.scenario_id, expected_artifact_id=artifact_id
    )
    if target.name != manifest["artifact_id"]:
        raise ValueError("Inference artifact directory does not match its identity.")
    return state, power, path
