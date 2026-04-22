import os
import json
import math
import random
import gc
from pathlib import Path
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

import model_unet as m

PATHS = {
    "data_root": Path("/kaggle/input/datasets/taisiyaglazova"),
    "encoder_checkpoint": Path("/kaggle/input/datasets/taisiyaglazova/ssl-full-encoder-best/encoder_best.pt"),
    "results_root": Path("/kaggle/working/stage5_results"),
}

# Пути к датасетам
DATASETS = {
    "bigp3_train": PATHS["data_root"] / "bigp3bci-downstream-train",
    "bigp3_benchmark": PATHS["data_root"] / "bigp3bci-downstream-benchmark",
    "bcicomp3": PATHS["data_root"] / "bcicompiii-ds2",  
}

GROUPS = ["train", "benchmark", "bcicomp3"]

SCENARIO_CONFIGS = {
    "full_ft": {
        "description": "Train full encoder + head",
        "use_discriminative_lr": False,
        "trainable_mode": "full",
        "use_warmup": False,
    },
    "low_lr_encoder": {
        "description": "Train full encoder + head with lower LR for encoder",
        "use_discriminative_lr": True,
        "trainable_mode": "full",
        "use_warmup": False,
    },
    "partial_ft": {
        "description": "Train only down4 + head",
        "use_discriminative_lr": True,
        "trainable_mode": "down4_only",
        "use_warmup": False,
    },
    "warmup": {
        "description": "Stage 1: head only, Stage 2: full FT with lower encoder LR",
        "use_discriminative_lr": True,
        "trainable_mode": "full",
        "use_warmup": True,
    },
}

# Воспроизводимость
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Загрузка эпох
def load_subject_epochs(subject_id: str, group: str):
    path = get_epochs_path(subject_id, group)
    if not path.exists():
        raise FileNotFoundError(f"Epochs file not found: {path}")

    data = np.load(path, allow_pickle=True)

    if "X" not in data or "y" not in data:
        raise KeyError(f"{path} must contain keys 'X' and 'y'. Found: {list(data.keys())}")

    X = data["X"]
    y = data["y"]

    return X, y
    
# Загрузка split
def load_subject_split(subject_id: str, group: str):
    path = get_split_path(subject_id, group)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        split = json.load(f)

    # Приводим формат bcicomp3 к стандартному формату pipeline
    if group == "bcicomp3":
        if "indices" not in split:
            raise KeyError("bcicomp3 split must contain top-level key 'indices'")

        idx = split["indices"]

        normalized_split = {
            "calib_pool_idx": idx["calib_pool_idx"],
            "test_rest_idx": idx["test_rest_idx"],
            "calib_idx": idx["calib_idx"],
        }
        return normalized_split

    return split
    
# Загрузка stats
def load_subject_stats(subject_id: str, p: int, group: str):
    path = get_stats_path(subject_id, p, group)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    data = np.load(path, allow_pickle=True)

    if "mean" not in data or "std" not in data:
        raise KeyError(f"{path} must contain keys 'mean' and 'std'. Found: {list(data.keys())}")

    mean = data["mean"]
    std = data["std"]

    return mean, std
    
# Объединённая загрузка bundle
def load_subject_bundle(subject_id: str, p: int, group: str):
    X, y = load_subject_epochs(subject_id, group)
    split = load_subject_split(subject_id, group)

    mean, std = (None, None)
    if p > 0:
        mean, std = load_subject_stats(subject_id, p, group)

    bundle = {
        "subject_id": subject_id,
        "group": group,
        "p": p,
        "X": X,
        "y": y,
        "split": split,
        "mean": mean,
        "std": std,
    }
    return bundle
    
# Функции для извлечения индексов
def get_test_indices(split: dict) -> np.ndarray:
    """
    Возвращает индексы test_rest в глобальной индексации subject-level X/y.
    """
    if "test_rest_idx" not in split:
        raise KeyError("split must contain 'test_rest_idx'")
    return np.asarray(split["test_rest_idx"], dtype=np.int64)


def get_calib_pool_indices(split: dict) -> np.ndarray:
    """
    Возвращает индексы calib_pool в глобальной индексации subject-level X/y.
    """
    if "calib_pool_idx" not in split:
        raise KeyError("split must contain 'calib_pool_idx'")
    return np.asarray(split["calib_pool_idx"], dtype=np.int64)


def get_calib_indices(split: dict, p: int) -> np.ndarray:
    """
    Возвращает индексы calib_p в глобальной индексации subject-level X/y.

    Для p=0 возвращает пустой массив.
    """
    if p == 0:
        return np.asarray([], dtype=np.int64)

    if "calib_idx" not in split:
        raise KeyError("split must contain 'calib_idx'")

    calib_idx_dict = split["calib_idx"]

    if not isinstance(calib_idx_dict, dict):
        raise TypeError(f"split['calib_idx'] must be dict, got {type(calib_idx_dict)}")

    if str(p) not in calib_idx_dict:
        raise KeyError(
            f"p={p} not found in split['calib_idx']; available keys: {list(calib_idx_dict.keys())}"
        )

    return np.asarray(calib_idx_dict[str(p)], dtype=np.int64)


def make_train_val_split(
    calib_idx: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
):
    """
    Делит calib_idx на train_idx и val_idx.

    Все индексы остаются глобальными относительно X/y конкретного subject.
    """
    calib_idx = np.asarray(calib_idx, dtype=np.int64)

    if len(calib_idx) == 0:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )

    if len(calib_idx) < 2:
        raise ValueError("calib_idx must contain at least 2 samples")

    y_calib = y[calib_idx]
    stratify_labels = y_calib if stratify else None

    try:
        train_idx, val_idx = train_test_split(
            calib_idx,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
    except ValueError as e:
        print(f"[WARN] Stratified split failed: {e}")
        print("[WARN] Falling back to non-stratified split.")
        train_idx, val_idx = train_test_split(
            calib_idx,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
    )


def prepare_run_indices(
    split: dict,
    y: np.ndarray,
    p: int,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Готовит все индексы для одного запуска.

    Возвращает словарь:
    - calib_pool_idx
    - calib_idx
    - train_idx
    - val_idx
    - test_idx
    """
    calib_pool_idx = get_calib_pool_indices(split)
    test_idx = get_test_indices(split)
    calib_idx = get_calib_indices(split, p)

    calib_pool_set = set(calib_pool_idx.tolist())
    test_set = set(test_idx.tolist())
    calib_set = set(calib_idx.tolist())

    # test и calib_pool не должны пересекаться
    if len(calib_pool_set & test_set) > 0:
        raise ValueError("calib_pool_idx intersects with test_rest_idx")

    # calib_p должен быть подмножеством calib_pool
    if len(calib_set) > 0 and not calib_set.issubset(calib_pool_set):
        raise ValueError("calib_idx is not a subset of calib_pool_idx")

    if p == 0:
        train_idx = np.asarray([], dtype=np.int64)
        val_idx = np.asarray([], dtype=np.int64)
    else:
        train_idx, val_idx = make_train_val_split(
            calib_idx=calib_idx,
            y=y,
            val_ratio=val_ratio,
            seed=seed,
            stratify=True,
        )

    return {
        "calib_pool_idx": calib_pool_idx,
        "calib_idx": calib_idx,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    
# Фукции нормализации
def safe_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Поканальная z-нормализация массива X формы (N, C, L)
    по mean/std формы (C,).
    """
    X = np.asarray(X, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"X must have shape (N, C, L), got {X.shape}")
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(f"mean/std must have shape (C,), got mean={mean.shape}, std={std.shape}")
    if X.shape[1] != len(mean) or X.shape[1] != len(std):
        raise ValueError(
            f"Channel mismatch: X has C={X.shape[1]}, mean={len(mean)}, std={len(std)}"
        )

    std_safe = np.maximum(std, eps)
    X_norm = (X - mean[None, :, None]) / std_safe[None, :, None]
    return X_norm.astype(np.float32)


def get_effective_stats(bundle: dict, subject_id: str, group: str, p: int, fallback_p_for_zero: int = 10):
    """
    Возвращает mean/std для данного запуска.
    
    Для p>0 используются stats именно этого p.
    Для p=0 используются fallback-статистики, например p=10.
    """
    if p > 0:
        mean = bundle["mean"]
        std = bundle["std"]
    else:
        mean, std = load_subject_stats(subject_id, fallback_p_for_zero, group)

    if mean is None or std is None:
        raise ValueError(f"Stats are not available for subject={subject_id}, group={group}, p={p}")

    return mean, std
    
# Функции нарезки по индексам
def slice_by_indices(X: np.ndarray, y: np.ndarray, idx: np.ndarray):
    """
    Возвращает подмножество X/y по глобальным индексам subject-level массива.
    """
    idx = np.asarray(idx, dtype=np.int64)

    if len(idx) == 0:
        X_empty = np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32)
        y_empty = np.empty((0,), dtype=y.dtype)
        return X_empty, y_empty

    return X[idx], y[idx]


def prepare_indexed_arrays(
    bundle: dict,
    indices_dict: dict,
    fallback_p_for_zero: int = 10,
):
    """
    Подготавливает нормализованные массивы:
    - X_train, y_train
    - X_val, y_val
    - X_test, y_test

    bundle должен содержать:
    - subject_id
    - group
    - p
    - X
    - y
    - mean/std (если p>0)
    """
    subject_id = bundle["subject_id"]
    group = bundle["group"]
    p = bundle["p"]

    X = bundle["X"]
    y = bundle["y"]

    mean, std = get_effective_stats(
        bundle=bundle,
        subject_id=subject_id,
        group=group,
        p=p,
        fallback_p_for_zero=fallback_p_for_zero,
    )

    X_norm = safe_standardize(X, mean, std)

    X_train, y_train = slice_by_indices(X_norm, y, indices_dict["train_idx"])
    X_val, y_val = slice_by_indices(X_norm, y, indices_dict["val_idx"])
    X_test, y_test = slice_by_indices(X_norm, y, indices_dict["test_idx"])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "mean": mean,
        "std": std,
    }
    
class EEGDataset(Dataset):
    """
    Простой Dataset для EEG-эпох.
    X: np.ndarray формы (N, C, L)
    y: np.ndarray формы (N,)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(f"X must have shape (N, C, L), got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must have shape (N,), got {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: len(X)={len(X)}, len(y)={len(y)}")

        self.X = X
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])      # float32, shape (C, L)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Создаёт DataLoader для заданных X/y.
    Если массив пустой, возвращает None.
    """
    if len(y) == 0:
        return None

    dataset = EEGDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader
    
def build_loaders(
    arrays_dict: dict,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    По словарю prepared arrays создаёт:
    - train_loader
    - val_loader
    - test_loader

    Для p=0 train/val будут None.
    """
    train_loader = make_loader(
        X=arrays_dict["X_train"],
        y=arrays_dict["y_train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = make_loader(
        X=arrays_dict["X_val"],
        y=arrays_dict["y_val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = make_loader(
        X=arrays_dict["X_test"],
        y=arrays_dict["y_test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }

# Универсальные функции путей
def get_epochs_path(subject_id: str, group: str) -> Path:
    assert group in GROUPS, f"Unknown group: {group}"

    if group == "train":
        path = DATASETS["bigp3_train"] / "train" / f"{subject_id}.npz"
    elif group == "benchmark":
        path = DATASETS["bigp3_benchmark"]/ "benchmark" / f"{subject_id}.npz"
    elif group == "bcicomp3":
        path = DATASETS["bcicomp3"] / "epochs" / f"{subject_id}_train_epochs_v1.npz"
    
    return path


def get_split_path(subject_id: str, group: str) -> Path:
    assert group in GROUPS, f"Unknown group: {group}"

    if group == "train":
        path = DATASETS["bigp3_train"] / "splits" / "train" / f"{subject_id}.json"
    elif group == "benchmark":
        path = DATASETS["bigp3_benchmark"] / "splits" / "benchmark" / f"{subject_id}.json"
    elif group == "bcicomp3":
        path = DATASETS["bcicomp3"] / "splits" / f"{subject_id}_time30_seed42_v1.json"
        
    return path


def get_stats_path(subject_id: str, p: int, group: str) -> Path:
    assert group in GROUPS, f"Unknown group: {group}"

    if group == "train":
        path = DATASETS["bigp3_train"] / "stats" / "train" / f"{subject_id}_p{p}.npz"
    elif group == "benchmark":
        path = DATASETS["bigp3_benchmark"] / "stats" / "benchmark" / f"{subject_id}_p{p}.npz"
    elif group == "bcicomp3":
        path = DATASETS["bcicomp3"] / "stats" / f"{subject_id}_time30_seed42_p{p}_v1.npz"
        
    return path

# Загрузка весов
def load_encoder_checkpoint_into_model_encoder(model_encoder: nn.Module, encoder_checkpoint: str, device: str = "cpu"):
    """
    Загружает encoder_best.pt в encoder downstream-модели.

    Ожидаемый формат checkpoint:
    {
        'inc': state_dict(...),
        'down1': state_dict(...),
        'down2': state_dict(...),
        'down3': state_dict(...),
        'down4': state_dict(...),
    }
    """
    ckpt = torch.load(encoder_checkpoint, map_location=device)

    expected_keys = ["inc", "down1", "down2", "down3", "down4"]
    missing = [k for k in expected_keys if k not in ckpt]
    if len(missing) > 0:
        raise KeyError(f"Encoder checkpoint is missing keys: {missing}. Found keys: {list(ckpt.keys())}")

    model_encoder.inc.load_state_dict(ckpt["inc"], strict=True)
    model_encoder.down1.load_state_dict(ckpt["down1"], strict=True)
    model_encoder.down2.load_state_dict(ckpt["down2"], strict=True)
    model_encoder.down3.load_state_dict(ckpt["down3"], strict=True)
    model_encoder.down4.load_state_dict(ckpt["down4"], strict=True)

    return model_encoder


# ============================================================
# FT scenario helpers: freeze / unfreeze / optimizer groups
# ============================================================


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Set requires_grad for all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = flag


def freeze_all(model: nn.Module) -> None:
    """Freeze all model parameters."""
    for p in model.parameters():
        p.requires_grad = False


def apply_trainable_mode(model: nn.Module, trainable_mode: str) -> None:
    """
    Configure which parts of the model are trainable.

    Expected model structure:
        model.encoder.inc
        model.encoder.down1
        model.encoder.down2
        model.encoder.down3
        model.encoder.down4
        model.head
    """
    freeze_all(model)

    if trainable_mode == "full":
        set_requires_grad(model.encoder, True)
        set_requires_grad(model.head, True)

    elif trainable_mode == "down4_only":
        set_requires_grad(model.encoder.down4, True)
        set_requires_grad(model.head, True)

    elif trainable_mode == "head_only":
        set_requires_grad(model.head, True)

    else:
        raise ValueError(f"Unknown trainable_mode: {trainable_mode}")


def get_trainable_parameter_names(model: nn.Module):
    """Return names of trainable parameters."""
    return [name for name, p in model.named_parameters() if p.requires_grad]


def count_parameters(model: nn.Module):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_ft_optimizer(
    model: nn.Module,
    scenario_name: str,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
):
    """
    Build AdamW optimizer according to FT scenario config.

    Scenarios:
        - full_ft
        - low_lr_encoder
        - partial_ft
        - warmup
    """
    scenario_cfg = SCENARIO_CONFIGS[scenario_name]

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    if not encoder_params and not head_params:
        raise ValueError("No trainable parameters found.")

    if scenario_cfg["use_discriminative_lr"]:
        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": lr_encoder,
                "weight_decay": weight_decay,
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay,
            })
        optimizer = torch.optim.AdamW(param_groups)

    else:
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            all_trainable,
            lr=lr_head,
            weight_decay=weight_decay,
        )

    return optimizer


def summarize_trainable_parameters(model: nn.Module, max_lines: int = 50) -> None:
    """Print a short summary of trainable parameters."""
    total, trainable = count_parameters(model)
    names = get_trainable_parameter_names(model)

    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {total - trainable:,}")
    print(f"Trainable tensors: {len(names)}")

    preview = names[:max_lines]
    if preview:
        print("\nTrainable parameter names:")
        for name in preview:
            print(" -", name)

    if len(names) > max_lines:
        print(f"... and {len(names) - max_lines} more")


def train_one_epoch(model, loader, optimizer, criterion, device: str):
    """
    Одна эпоха обучения.
    Возвращает средний loss по эпохе.
    """
    if loader is None:
        raise ValueError("train loader is None")

    model.train()

    running_loss = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(n_samples, 1)
    return epoch_loss


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device: str):
    """
    Одна эпоха валидации.
    Возвращает средний loss по эпохе.
    """
    if loader is None:
        raise ValueError("val loader is None")

    model.eval()

    running_loss = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(n_samples, 1)
    return epoch_loss


# training history
def init_history():
    return {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }


def append_history(history: dict, epoch: int, train_loss: float, val_loss: float):
    history["epoch"].append(epoch)
    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    return history

# цикл обучения
def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device: str,
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0,
    verbose: bool = True,
):
    """
    Train/val цикл с early stopping по val_loss.

    Возвращает словарь с:
    - history
    - best_epoch
    - best_val_loss
    - best_state_dict
    - stopped_epoch
    """
    if train_loader is None:
        raise ValueError("train_loader is None")
    if val_loader is None:
        raise ValueError("val_loader is None")

    history = init_history()
    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    stopped_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        append_history(history, epoch, train_loss, val_loss)

        early_stopper.step(val_loss, model, epoch)

        if verbose:
            msg = (
                f"epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"best_val={early_stopper.best_value:.6f} @ epoch {early_stopper.best_epoch}"
            )
            print(msg)

        if early_stopper.should_stop:
            stopped_epoch = epoch
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}.")
            break

    result = {
        "history": history,
        "best_epoch": early_stopper.best_epoch,
        "best_val_loss": early_stopper.best_value,
        "best_state_dict": early_stopper.best_state_dict,
        "stopped_epoch": stopped_epoch,
    }
    return result

# Optimizer и Loss
def build_criterion(y_train=None, device="cpu"):
    if y_train is None:
        return nn.CrossEntropyLoss()

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    weight_pos = n_neg / (n_pos + 1e-8)

    class_weights = torch.tensor([1.0, weight_pos], dtype=torch.float32)

    return nn.CrossEntropyLoss(weight=class_weights.to(device))


# Загрузка best state
def load_best_model_state(model: nn.Module, fit_result: dict):
    """
    Загружает best_state_dict обратно в модель.
    """
    best_state_dict = fit_result.get("best_state_dict", None)
    if best_state_dict is None:
        raise ValueError("fit_result does not contain 'best_state_dict'")
    model.load_state_dict(best_state_dict)
    return model


class EarlyStopping:
    """
    Early stopping по val_loss.

    mode='min' означает, что метрика должна уменьшаться.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode

        self.best_value = None
        self.best_epoch = None
        self.best_state_dict = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        else:
            return value > (self.best_value + self.min_delta)

    def step(self, value: float, model: nn.Module, epoch: int):
        """
        Обновляет состояние early stopping после очередной эпохи.
        """
        if self._is_improvement(value):
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.best_state_dict = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# prediction на loader
@torch.no_grad()
def predict_on_loader(model, loader, device: str):
    """
    Прогоняет модель по loader и возвращает:
    - y_true
    - prob_score   : probability of positive class
    - logit_score  : raw logit of positive class
    - y_pred       : threshold=0.5 on prob_score
    """
    if loader is None:
        raise ValueError("loader is None")

    model.eval()

    all_y = []
    all_prob = []
    all_logit = []
    all_pred = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        logits = model(xb)                    # (B, 2)
        probs = torch.softmax(logits, dim=1)  # (B, 2)

        prob_score = probs[:, 1]
        logit_score = logits[:, 1]
        pred = (prob_score >= 0.5).long()

        all_y.append(yb.numpy())
        all_prob.append(prob_score.cpu().numpy())
        all_logit.append(logit_score.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_y) if len(all_y) > 0 else np.array([], dtype=np.int64)
    prob_score = np.concatenate(all_prob) if len(all_prob) > 0 else np.array([], dtype=np.float32)
    logit_score = np.concatenate(all_logit) if len(all_logit) > 0 else np.array([], dtype=np.float32)
    y_pred = np.concatenate(all_pred) if len(all_pred) > 0 else np.array([], dtype=np.int64)

    return {
        "y_true": y_true,
        "prob_score": prob_score,
        "logit_score": logit_score,
        "y_pred": y_pred,
    }


## FDR
def compute_fisher_fdr(y_true: np.ndarray, score: np.ndarray):
    """
    Fisher's Discriminant Ratio:
        FDR = (mu_pos - mu_neg)^2 / (var_pos + var_neg)

    score должен быть непрерывным classifier score.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    pos_scores = score[y_true == 1]
    neg_scores = score[y_true == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan

    mu_pos = np.mean(pos_scores)
    mu_neg = np.mean(neg_scores)

    var_pos = np.var(pos_scores)
    var_neg = np.var(neg_scores)

    denom = var_pos + var_neg
    if denom <= 0:
        return np.nan

    return float((mu_pos - mu_neg) ** 2 / denom)


# accuracy, f1, precision, recall, fdr
def compute_metrics(
    y_true: np.ndarray,
    prob_score: np.ndarray,
    logit_score: np.ndarray,
    y_pred: np.ndarray,
):
    """
    Считает метрики в соответствии с логикой статьи:
    - AUC: по probability score
    - Accuracy: по binary predictions
    - F1: по binary predictions
    - Precision/Recall: по binary predictions
    - FDR: Fisher's Discriminant Ratio по classifier score
    """
    y_true = np.asarray(y_true).astype(int)
    prob_score = np.asarray(prob_score).astype(float)
    logit_score = np.asarray(logit_score).astype(float)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {}

    if len(np.unique(y_true)) < 2:
        metrics["auc"] = np.nan
    else:
        metrics["auc"] = float(roc_auc_score(y_true, prob_score))

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["fdr"] = compute_fisher_fdr(y_true, logit_score)

    return metrics


# Объединяющая функия test evaluation
def evaluate_on_test(model, test_loader, device: str):
    pred_dict = predict_on_loader(
        model=model,
        loader=test_loader,
        device=device,
    )

    metrics = compute_metrics(
        y_true=pred_dict["y_true"],
        prob_score=pred_dict["prob_score"],
        logit_score=pred_dict["logit_score"],
        y_pred=pred_dict["y_pred"],
    )

    return {
        "predictions": pred_dict,
        "metrics": metrics,
    }

def select_best_threshold(y_true, prob_score, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.40, 71)

    best_t = 0.5
    best_f1 = -1.0

    for t in grid:
        y_pred = (prob_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


# Подсчёт метрик с заданным порогом
def compute_metrics_with_threshold(y_true, prob_score, logit_score, threshold):
    y_pred = (prob_score >= threshold).astype(int)
    return compute_metrics(
        y_true=y_true,
        prob_score=prob_score,
        logit_score=logit_score,
        y_pred=y_pred,
    )


def train_standard_ft(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    scenario_name: str,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
):
    """
    Standard FT training for:
        - full_ft
        - low_lr_encoder
        - partial_ft
    """
    scenario_cfg = SCENARIO_CONFIGS[scenario_name]

    apply_trainable_mode(model, scenario_cfg["trainable_mode"])

    optimizer = build_ft_optimizer(
        model=model,
        scenario_name=scenario_name,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )

    stopper = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    history_rows = []
    stopped_epoch = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history_rows.append({
            "stage": "main",
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        stopper.step(val_loss, model, epoch)

        if stopper.should_stop:
            stopped_epoch = epoch
            break

    if stopped_epoch == 0:
        stopped_epoch = max_epochs

    return {
        "best_model_state": stopper.best_state_dict,
        "history_rows": history_rows,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_value,
        "stopped_epoch": stopped_epoch,
    }


def train_warmup_ft(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
    warmup_epochs: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
):
    """
    Two-stage FT training:
        Stage 1: head-only warmup
        Stage 2: joint FT (full encoder + head with discriminative LR)
    """
    history_rows = []

    # -------------------------
    # Stage 1: head-only warmup
    # -------------------------
    apply_trainable_mode(model, "head_only")

    warmup_optimizer = build_ft_optimizer(
        model=model,
        scenario_name="warmup",
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )

    for epoch in range(1, warmup_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=warmup_optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history_rows.append({
            "stage": "warmup",
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

    # --------------------------------------
    # Stage 2: joint FT with early stopping
    # --------------------------------------
    apply_trainable_mode(model, "full")

    main_optimizer = build_ft_optimizer(
        model=model,
        scenario_name="warmup",
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )

    stopper = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    stopped_epoch = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=main_optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history_rows.append({
            "stage": "main",
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        stopper.step(val_loss, model, epoch)

        if stopper.should_stop:
            stopped_epoch = epoch
            break

    if stopped_epoch == 0:
        stopped_epoch = max_epochs

    return {
        "best_model_state": stopper.best_state_dict,
        "history_rows": history_rows,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_value,
        "stopped_epoch": stopped_epoch,
    }


def history_rows_to_df(history_rows):
    """Convert accumulated history rows to DataFrame."""
    if not history_rows:
        return pd.DataFrame(columns=["stage", "epoch", "train_loss", "val_loss"])
    return pd.DataFrame(history_rows)


def extract_split_stats(indices_dict: dict, y: np.ndarray):
    def _count(idx):
        idx = np.asarray(idx, dtype=np.int64)
        n = len(idx)
        if n == 0:
            return {"n": 0, "n_pos": 0, "n_neg": 0}
        n_pos = int(y[idx].sum())
        n_neg = int(n - n_pos)
        return {"n": n, "n_pos": n_pos, "n_neg": n_neg}

    train_stats = _count(indices_dict["train_idx"])
    val_stats = _count(indices_dict["val_idx"])
    test_stats = _count(indices_dict["test_idx"])
    calib_stats = _count(indices_dict["calib_idx"])

    return {
        "n_calib": calib_stats["n"],
        "n_val": val_stats["n"],
        "n_test": test_stats["n"],
        "n_pos_calib": calib_stats["n_pos"],
        "n_pos_val": val_stats["n_pos"],
        "n_pos_test": test_stats["n_pos"],
    }


def run_one(
    subject_id: str,
    p: int,
    scenario: str,
    group: str,
    runtime_config: dict,
    scratch_lr: float = 1e-4,
    scratch_weight_decay: float = 1e-4,
    encoder_checkpoint: str = None,
    ft_strategy: str = None,
    lr_encoder: float = 1e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 3,
):
    """
    Один полный запуск pipeline для одного subject / p / scenario.

    Поддерживает:
    - scenario="scratch"
    - scenario="ssl_ft"

    Для scenario="ssl_ft" и p > 0 требуется ft_strategy:
    - "full_ft"
    - "low_lr_encoder"
    - "partial_ft"
    - "warmup"

    Для p > 0:
    - threshold подбирается на val по максимуму F1
    - затем применяется на test

    Возвращает словарь с метриками и служебной информацией.
    """
    seed = runtime_config["seed"]
    set_seed(seed)

    device = runtime_config["device"]
    val_ratio = runtime_config["val_ratio"]
    fallback_p_for_zero = runtime_config["fallback_p_for_zero"]
    batch_size = runtime_config["batch_size"]
    num_workers = runtime_config["num_workers"]
    pin_memory = runtime_config["pin_memory"]
    max_epochs = runtime_config["max_epochs"]
    patience = runtime_config["patience"]
    min_delta = runtime_config["min_delta"]

    if scenario not in {"scratch", "ssl_ft"}:
        raise ValueError(f"Unknown scenario: {scenario}")

    if scenario == "ssl_ft" and p > 0 and ft_strategy is None:
        raise ValueError("For scenario='ssl_ft' and p > 0, ft_strategy must be provided.")

    if scenario == "ssl_ft" and encoder_checkpoint is None:
        raise ValueError("For scenario='ssl_ft', encoder_checkpoint must be provided.")

    bundle = load_subject_bundle(subject_id=subject_id, p=p, group=group)

    indices = prepare_run_indices(
        split=bundle["split"],
        y=bundle["y"],
        p=p,
        val_ratio=val_ratio,
        seed=seed,
    )

    arrays = prepare_indexed_arrays(
        bundle=bundle,
        indices_dict=indices,
        fallback_p_for_zero=fallback_p_for_zero,
    )

    loaders = build_loaders(
        arrays_dict=arrays,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    # criterion
    criterion = build_criterion(arrays["y_train"], device=device) if p > 0 else build_criterion(device=device)

    split_stats = extract_split_stats(indices, bundle["y"])
    
    # model
    if scenario == "scratch":
        model = m.build_model(
            scenario="scratch",
            device=device,
        )

    elif scenario == "ssl_ft":
        model = m.build_model(
            scenario="ssl_ft",
            encoder_checkpoint=encoder_checkpoint,
            device=device,
        )
    
    # -------------------------------------------------
    # CASE A: p == 0 -> no training
    # -------------------------------------------------
    if p == 0:
        test_result = evaluate_on_test(
            model=model,
            test_loader=test_loader,
            device=device,
        )
    
        result_row = {
            "subject_id": subject_id,
            "group": group,
            "p": p,
            "scenario": scenario,
            "ft_strategy": ft_strategy if scenario == "ssl_ft" else None,
            "seed": seed,
            "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint is not None else None,
            "lr_encoder": lr_encoder if scenario == "ssl_ft" else None,
            "lr_head": lr_head if scenario == "ssl_ft" else None,
            "weight_decay": weight_decay if scenario == "ssl_ft" else None,
            "warmup_epochs": warmup_epochs if ft_strategy == "warmup" else None,
            "selected_threshold": 0.5,
            "val_f1_at_selected_threshold": None,
    
            **split_stats,
    
            "best_epoch": None,
            "best_val_loss": None,
            "stopped_epoch": None,
    
            "auc": test_result["metrics"]["auc"],
            "accuracy": test_result["metrics"]["accuracy"],
            "f1": test_result["metrics"]["f1"],
            "precision": test_result["metrics"]["precision"],
            "recall": test_result["metrics"]["recall"],
            "fdr": test_result["metrics"]["fdr"],
        }
    
        history_df = pd.DataFrame(columns=["stage", "epoch", "train_loss", "val_loss"])
    
        return {
            "result_row": result_row,
            "history_df": history_df,
            "predictions": test_result["predictions"],
        }

    # -------------------------------------------------
    # CASE B1: scratch -> old training logic
    # -------------------------------------------------
    if scenario == "scratch":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=scratch_lr,
            weight_decay=scratch_weight_decay,
        )

        fit_result = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            verbose=False,
        )

        model = load_best_model_state(model, fit_result)

        # ---------- val threshold selection ----------
        val_pred = predict_on_loader(
            model=model,
            loader=val_loader,
            device=device,
        )

        selected_threshold, val_f1_at_selected_threshold = select_best_threshold(
            y_true=val_pred["y_true"],
            prob_score=val_pred["prob_score"],
        )

        # ---------- test evaluation with selected threshold ----------
        test_pred = predict_on_loader(
            model=model,
            loader=test_loader,
            device=device,
        )

        test_metrics = compute_metrics_with_threshold(
            y_true=test_pred["y_true"],
            prob_score=test_pred["prob_score"],
            logit_score=test_pred["logit_score"],
            threshold=selected_threshold,
        )

        history_df = pd.DataFrame(fit_result["history"])
        if "stage" not in history_df.columns:
            history_df["stage"] = "main"
            history_df = history_df[["stage", "epoch", "train_loss", "val_loss"]]

        test_predictions = {
            "y_true": test_pred["y_true"],
            "prob_score": test_pred["prob_score"],
            "logit_score": test_pred["logit_score"],
            "y_pred": (test_pred["prob_score"] >= selected_threshold).astype(int),
        }

        result_row = {
            "subject_id": subject_id,
            "group": group,
            "p": p,
            "scenario": scenario,
            "ft_strategy": None,
            "seed": seed,
            "encoder_checkpoint": None,
            "lr_encoder": None,
            "lr_head": None,
            "weight_decay": weight_decay,
            "warmup_epochs": None,
            "selected_threshold": selected_threshold,
            "val_f1_at_selected_threshold": val_f1_at_selected_threshold,

            **split_stats,

            "best_epoch": fit_result["best_epoch"],
            "best_val_loss": fit_result["best_val_loss"],
            "stopped_epoch": fit_result["stopped_epoch"],

            "auc": test_metrics["auc"],
            "accuracy": test_metrics["accuracy"],
            "f1": test_metrics["f1"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "fdr": test_metrics["fdr"],
        }

        return {
            "result_row": result_row,
            "history_df": history_df,
            "predictions": test_predictions,
        }

    # -------------------------------------------------
    # CASE B2: ssl_ft -> Block 4 FT strategies
    # -------------------------------------------------
    if ft_strategy == "warmup":
        train_out = train_warmup_ft(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            lr_encoder=lr_encoder,
            lr_head=lr_head,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
        )
    else:
        train_out = train_standard_ft(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            scenario_name=ft_strategy,
            lr_encoder=lr_encoder,
            lr_head=lr_head,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
        )

    model.load_state_dict(train_out["best_model_state"])

    # ---------- val threshold selection ----------
    val_pred = predict_on_loader(
        model=model,
        loader=val_loader,
        device=device,
    )

    selected_threshold, val_f1_at_selected_threshold = select_best_threshold(
        y_true=val_pred["y_true"],
        prob_score=val_pred["prob_score"],
    )

    # ---------- test evaluation with selected threshold ----------
    test_pred = predict_on_loader(
        model=model,
        loader=test_loader,
        device=device,
    )

    test_metrics = compute_metrics_with_threshold(
        y_true=test_pred["y_true"],
        prob_score=test_pred["prob_score"],
        logit_score=test_pred["logit_score"],
        threshold=selected_threshold,
    )

    history_df = history_rows_to_df(train_out["history_rows"])

    test_predictions = {
        "y_true": test_pred["y_true"],
        "prob_score": test_pred["prob_score"],
        "logit_score": test_pred["logit_score"],
        "y_pred": (test_pred["prob_score"] >= selected_threshold).astype(int),
    }

    result_row = {
        "subject_id": subject_id,
        "group": group,
        "p": p,
        "scenario": scenario,
        "ft_strategy": ft_strategy,
        "seed": seed,
        "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint is not None else None,
        "lr_encoder": lr_encoder,
        "lr_head": lr_head,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs if ft_strategy == "warmup" else None,
        "selected_threshold": selected_threshold,
        "val_f1_at_selected_threshold": val_f1_at_selected_threshold,

        **split_stats,

        "best_epoch": train_out["best_epoch"],
        "best_val_loss": train_out["best_val_loss"],
        "stopped_epoch": train_out["stopped_epoch"],

        "auc": test_metrics["auc"],
        "accuracy": test_metrics["accuracy"],
        "f1": test_metrics["f1"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "fdr": test_metrics["fdr"],
    }

    return {
        "result_row": result_row,
        "history_df": history_df,
        "predictions": test_predictions,
    }

# Для сохранения путей
# Вспомогательная функция
def format_scientific(x: float) -> str:
    return f"{x:.0e}".replace("-", "m")

def make_run_tag(
    subject_id: str,
    group: str,
    p: int,
    scenario: str,
    seed: int,
    ft_strategy: str = None,
    lr_encoder: float = None,
    lr_head: float = None,
    weight_decay: float = None,
):
    parts = [
        f"{group}",
        f"{subject_id}",
        f"p{p}",
        f"{scenario}",
        f"seed{seed}",
    ]

    # --- FT часть (добавляется только если есть) ---
    if ft_strategy is not None:
        parts.append(ft_strategy)

    if lr_encoder is not None:
        parts.append(f"lre{format_scientific(lr_encoder)}")

    if lr_head is not None:
        parts.append(f"lrh{format_scientific(lr_head)}")

    if weight_decay is not None:
        parts.append(f"wd{format_scientific(weight_decay)}")

    return "__".join(parts)

def ensure_results_dirs(results_root: Path):
    results_root = Path(results_root)
    (results_root / "history").mkdir(parents=True, exist_ok=True)
    (results_root / "predictions").mkdir(parents=True, exist_ok=True)
    (results_root / "tables").mkdir(parents=True, exist_ok=True)
    return results_root


# Сохранение hystory и prediction
def save_history_df(history_df: pd.DataFrame, run_tag: str, results_root: Path):
    results_root = ensure_results_dirs(results_root)
    out_path = results_root / "history" / f"{run_tag}.csv"
    history_df.to_csv(out_path, index=False)
    return out_path


def save_predictions_npz(predictions: dict, run_tag: str, results_root: Path):
    results_root = ensure_results_dirs(results_root)
    out_path = results_root / "predictions" / f"{run_tag}.npz"

    np.savez_compressed(
        out_path,
        y_true=predictions["y_true"],
        prob_score=predictions["prob_score"],
        logit_score=predictions["logit_score"],
        y_pred=predictions["y_pred"],
    )
    return out_path


# Запуск одного run с сохранением
def run_one_and_save(
    subject_id: str,
    p: int,
    scenario: str,
    group: str,
    runtime_config: dict,
    results_root: Path,
    scratch_lr: float = None,
    scratch_weight_decay: float = None,
    encoder_checkpoint: str = None,
    ft_strategy: str = None,
    lr_encoder: float = None,
    lr_head: float = None,
    weight_decay: float = None,
    warmup_epochs: int = 3,
    save_history: bool = True,
    save_predictions: bool = True,
):
    """
    Универсальная обёртка над run_one(...), которая:
    - запускает эксперимент
    - сохраняет history
    - сохраняет predictions
    - возвращает result_row c путями к артефактам
    """
    results_root = ensure_results_dirs(results_root)
    seed = runtime_config["seed"]

    out = run_one(
        subject_id=subject_id,
        p=p,
        scenario=scenario,
        group=group,
        runtime_config=runtime_config,
        scratch_lr=scratch_lr,
        scratch_weight_decay=scratch_weight_decay,
        encoder_checkpoint=encoder_checkpoint,
        ft_strategy=ft_strategy,
        lr_encoder=lr_encoder if lr_encoder is not None else 1e-5,
        lr_head=lr_head if lr_head is not None else 1e-4,
        weight_decay=weight_decay if weight_decay is not None else 1e-4,
        warmup_epochs=warmup_epochs,
    )

    run_tag = make_run_tag(
        subject_id=subject_id,
        group=group,
        p=p,
        scenario=scenario,
        seed=seed,
        ft_strategy=ft_strategy,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )

    history_path = None
    pred_path = None

    history_df = out["history_df"]
    predictions = out["predictions"]
    result_row = dict(out["result_row"])

    if save_history and history_df is not None and len(history_df) > 0:
        history_path = save_history_df(
            history_df=history_df,
            run_tag=run_tag,
            results_root=results_root,
        )

    if save_predictions and predictions is not None:
        pred_path = save_predictions_npz(
            predictions=predictions,
            run_tag=run_tag,
            results_root=results_root,
        )

    result_row["run_tag"] = run_tag
    result_row["history_path"] = str(history_path) if history_path is not None else None
    result_row["predictions_path"] = str(pred_path) if pred_path is not None else None

    return result_row


def run_many(
    subject_list,
    p_list,
    scenario_list,
    group: str,
    runtime_config: dict,
    results_root: Path,
    scratch_lr: float = None,
    scratch_weight_decay: float = None,
    encoder_checkpoint: str = None,
    ft_strategy_list: list = None,
    param_grid: list = None,
    lr_encoder: float = 1e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 3,
    save_history: bool = True,
    save_predictions: bool = True,
    save_summary_every: int = 1,
    continue_on_error: bool = True,
):
    """
    Универсальный массовый запуск экспериментов.

    Поддерживает:
    - scratch
    - ssl_ft
    - фиксированные FT-гиперпараметры
    - grid search по param_grid
    """
    results_root = ensure_results_dirs(results_root)
    summary_path = results_root / "tables" / "summary_results.csv"

    seed = runtime_config["seed"]
    rows = []
    run_counter = 0
    expanded_runs = []

    # Если param_grid не задан, используем один "виртуальный" набор параметров
    effective_param_grid = param_grid if param_grid is not None else [{
        "lr_encoder": lr_encoder,
        "lr_head": lr_head,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
    }]

    for subject_id in subject_list:
        for p in p_list:
            for scenario in scenario_list:
                if scenario == "ssl_ft":
                    current_ft_list = ft_strategy_list if ft_strategy_list is not None else [None]

                    for ft_strategy in current_ft_list:
                        for params in effective_param_grid:
                            expanded_runs.append({
                                "subject_id": subject_id,
                                "p": p,
                                "scenario": scenario,
                                "ft_strategy": ft_strategy,
                                "lr_encoder": params["lr_encoder"],
                                "lr_head": params["lr_head"],
                                "weight_decay": params["weight_decay"],
                                "warmup_epochs": params.get("warmup_epochs", 3),
                            })
                else:
                    expanded_runs.append({
                        "subject_id": subject_id,
                        "p": p,
                        "scenario": scenario,
                        "ft_strategy": None,
                        "lr_encoder": None,
                        "lr_head": None,
                        "weight_decay": None,
                        "warmup_epochs": None,
                    })

    total_runs = len(expanded_runs)
    print(f"Planned runs: {total_runs}")

    for run_cfg in expanded_runs:
        subject_id = run_cfg["subject_id"]
        p = run_cfg["p"]
        scenario = run_cfg["scenario"]
        ft_strategy = run_cfg["ft_strategy"]
        run_lr_encoder = run_cfg["lr_encoder"]
        run_lr_head = run_cfg["lr_head"]
        run_weight_decay = run_cfg["weight_decay"]
        run_warmup_epochs = run_cfg["warmup_epochs"]

        run_counter += 1
        print(
            f"[{run_counter}/{total_runs}] "
            f"subject={subject_id} | group={group} | p={p} | "
            f"scenario={scenario} | ft_strategy={ft_strategy} | seed={seed} | "
            f"lr_enc={run_lr_encoder} | lr_head={run_lr_head} | wd={run_weight_decay}"
        )

        try:
            row = run_one_and_save(
                subject_id=subject_id,
                p=p,
                scenario=scenario,
                group=group,
                runtime_config=runtime_config,
                results_root=results_root,
                scratch_lr=scratch_lr,
                scratch_weight_decay=scratch_weight_decay,
                encoder_checkpoint=encoder_checkpoint if scenario == "ssl_ft" else None,
                ft_strategy=ft_strategy,
                lr_encoder=run_lr_encoder if run_lr_encoder is not None else lr_encoder,
                lr_head=run_lr_head if run_lr_head is not None else lr_head,
                weight_decay=run_weight_decay if run_weight_decay is not None else weight_decay,
                warmup_epochs=run_warmup_epochs if run_warmup_epochs is not None else warmup_epochs,
                save_history=save_history,
                save_predictions=save_predictions,
            )
            row["status"] = "ok"
            row["error"] = None

        except Exception as e:
            if not continue_on_error:
                raise

            row = {
                "subject_id": subject_id,
                "group": group,
                "p": p,
                "scenario": scenario,
                "ft_strategy": ft_strategy,
                "seed": seed,
                "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint is not None else None,
                "lr_encoder": run_lr_encoder if scenario == "ssl_ft" else None,
                "lr_head": run_lr_head if scenario == "ssl_ft" else None,
                "weight_decay": run_weight_decay if scenario == "ssl_ft" else None,
                "warmup_epochs": run_warmup_epochs if ft_strategy == "warmup" else None,
                "status": "error",
                "error": repr(e),
            }
            print(f"[ERROR] {row['error']}")

        rows.append(row)

        if run_counter % save_summary_every == 0:
            pd.DataFrame(rows).to_csv(summary_path, index=False)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(summary_path, index=False)
    return results_df

# Функция для чтения summary
def load_summary_results(results_root: Path):
    summary_path = Path(results_root) / "tables" / "summary_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return pd.read_csv(summary_path)

