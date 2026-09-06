"""Inference-only TabM H5 loader for the production bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config: dict) -> dict:
    from final_solution.tabm_h3.features import FEATURES

    if config.get("schema_version") != 1:
        raise ValueError("Only schema version 1 is supported")
    if config.get("train_horizon") != 5 or config.get("corridor") != "KZT":
        raise ValueError("This runtime supports only KZT NOW H5")
    if config.get("features") != FEATURES:
        raise ValueError("Feature names/order differ from the production FULL33 contract")
    calibration = config["calibration"]
    if calibration["method"] != "identity":
        values = [calibration.get("intercept"), calibration.get("slope")]
        if calibration["method"] != "prior_year_monotone_platt" or not np.isfinite(values).all() or calibration["slope"] <= 0:
            raise ValueError("Invalid monotone calibration")
    policy = config["policy"]
    if policy.get("kind") != "rank" or policy.get("window") != 63:
        raise ValueError("H5 production bundle requires a causal 63-session rank policy")
    return config


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path)
    return validate_config(json.loads(path.read_text(encoding="utf-8"))), path.parent


class Predictor:
    def __init__(self, config: dict, base: str | Path):
        import torch
        from tabm import TabM
        from rtdl_num_embeddings import PeriodicEmbeddings

        self.config = validate_config(config)
        self.base = Path(base)
        self.features = config["features"]
        torch.set_num_threads(1)
        weights = self.base / config["model"]["weights"]
        preprocessor = self.base / config["model"]["preprocessor"]
        if sha(weights) != config["model"]["weights_sha256"] or sha(preprocessor) != config["model"]["preprocessor_sha256"]:
            raise ValueError("Model artifact checksum mismatch")
        self.pre = joblib.load(preprocessor)
        n_features = 2 * len(self.features)
        self.model = TabM.make(
            n_num_features=n_features,
            cat_cardinalities=[5],
            d_out=1,
            num_embeddings=PeriodicEmbeddings(n_features, **config["numerical_embeddings"]),
            **config["architecture"],
        )
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, frame):
        import torch

        if not frame.corridor.eq("KZT").all():
            raise ValueError("KZT-only model cannot score other corridors")
        missing = set(self.features) - set(frame)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        values = frame[self.features]
        x = np.concatenate([self.pre.transform(values), values.isna().to_numpy(float)], axis=1).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite transformed features")
        # KZT is category code 2 in the frozen five-corridor encoding.
        categories = np.full((len(x), 1), 2, dtype=np.int64)
        parts = []
        with torch.inference_mode():
            for start in range(0, len(x), 512):
                logits = self.model(torch.from_numpy(x[start:start + 512]), torch.from_numpy(categories[start:start + 512]))
                parts.append(logits.squeeze(-1).sigmoid().mean(dim=1).numpy())
        raw = np.concatenate(parts).astype(np.float64) if parts else np.array([], dtype=np.float64)
        calibration = self.config["calibration"]
        if calibration["method"] == "identity":
            return raw, raw.copy()
        from scipy.special import expit

        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        probability = expit(calibration["intercept"] + calibration["slope"] * np.log(clipped / (1 - clipped)))
        return raw, probability
