"""Train and evaluate the unified 116-class FSL v4 (322-dim, 16-step, 128-hidden GRU)
classifier combining:
- FSL-105 (`fsl105_keypoints/fsl105_clips.npz`, 104 classes excluding ID 2),
- Clean greeting keypoints (`greeting_keypoints/`), and
- Freshly extracted keypoints from raw `coordinates/videos` (`transaction_keypoints/transaction_clips.npz`,
  26 classes: 14 shared + 12 new classes).

Also rebuilds the pose-free fallback (`assets/ml/transaction_signs.json`) strictly from
freshly extracted hand keypoints in `transaction_clips.npz` (never reading `coordinates/*.npy`)
while leaving the original baseline `assets/ml/fsl105_signs.json` and `assets/ml/fsl105_labels.json`
untouched on disk.

Outputs:
- `assets/ml/fsl_combined_signs.json`
- `assets/ml/fsl_combined_labels.json`
- `assets/ml/transaction_signs.json`
- `transaction_keypoints/evaluation_metrics.json`
- `transaction_keypoints/confusion_matrix.csv`
- `transaction_keypoints/test_samples.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

import train_fsl105 as t105
import train_signs as ts_v1

TX_CLIPS_PATH = ROOT / "transaction_keypoints" / "transaction_clips.npz"
TX_SUMMARY_PATH = ROOT / "transaction_keypoints" / "dataset_summary.json"
COMBINED_MODEL_PATH = ROOT / "assets" / "ml" / "fsl_combined_signs.json"
COMBINED_LABELS_PATH = ROOT / "assets" / "ml" / "fsl_combined_labels.json"
TX_FALLBACK_MODEL_PATH = ROOT / "assets" / "ml" / "transaction_signs.json"
METRICS_OUT_PATH = ROOT / "transaction_keypoints" / "evaluation_metrics.json"
CONFUSION_OUT_PATH = ROOT / "transaction_keypoints" / "confusion_matrix.csv"
SAMPLES_OUT_PATH = ROOT / "transaction_keypoints" / "test_samples.json"


def train_gru_unified(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    num_classes: int,
    epochs: int = 75,
    wd: float = 2.5e-4,
    feat_drop: float = 0.12,
    smooth: float = 0.04,
) -> tuple[dict[str, np.ndarray], int]:
    """Train v4 mirror-augmented GRU with validation checkpointing in the fine-tuning phase."""
    rng = np.random.default_rng(37)

    def init_w(a: int, b: int) -> np.ndarray:
        return rng.normal(0.0, math.sqrt(2.0 / (a + b)), (a, b)).astype("f")

    p = {
        "w": init_w(t105.FEATURES, 3 * t105.HIDDEN),
        "u": init_w(t105.HIDDEN, 3 * t105.HIDDEN),
        "b": np.zeros(3 * t105.HIDDEN, "f"),
        "out": init_w(t105.HIDDEN + t105.FEATURES, num_classes),
        "bias": np.zeros(num_classes, "f"),
    }
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(t) for k, t in p.items()}
    step = 0
    best_key = (-1.0, -999.0)
    best_epoch = 0
    best_p: dict[str, np.ndarray] | None = None

    for epoch in range(epochs):
        indices = rng.permutation(len(x_tr))
        for start in range(0, len(x_tr), 64):
            batch = indices[start : start + 64]
            a = x_tr[batch].copy()
            a[::2] = t105.mirror_v3(a[::2])
            if feat_drop > 0:
                mask = (
                    rng.random((a.shape[0], 1, a.shape[2])) >= feat_drop
                ).astype("f") / (1.0 - feat_drop)
                a *= mask

            logits, (summary, history) = t105.forward_v4(a, p)
            prob = np.exp(logits - logits.max(axis=1, keepdims=True))
            prob = prob / prob.sum(axis=1, keepdims=True)
            d = prob - (smooth / num_classes)
            d[np.arange(len(d)), y_tr[batch]] -= 1.0 - smooth
            d /= len(d)

            g = {k: np.zeros_like(t) for k, t in p.items()}
            g["out"] = summary.T @ d
            g["bias"] = d.sum(axis=0)
            dh = (d @ p["out"].T)[:, : t105.HIDDEN]
            dx_all = np.empty((len(a), t105.STEPS, 3 * t105.HIDDEN), dtype="f")
            for i in range(t105.STEPS - 1, -1, -1):
                prev, z, r, n, un = history[i]
                dn = dh * (1.0 - z) * (1.0 - n * n)
                dz = dh * (prev - n) * z * (1.0 - z)
                dr = dn * un * r * (1.0 - r)
                dx_all[:, i, : t105.HIDDEN] = dz
                dx_all[:, i, t105.HIDDEN : 2 * t105.HIDDEN] = dr
                dx_all[:, i, 2 * t105.HIDDEN :] = dn
                du = np.concatenate([dz, dr, dn * r], axis=1)
                g["u"] += prev.T @ du
                dh = dh * z + du @ p["u"].T
            g["w"] = a.reshape(-1, t105.FEATURES).T @ dx_all.reshape(
                -1, 3 * t105.HIDDEN
            )
            g["b"] = dx_all.sum(axis=(0, 1))

            step += 1
            lr = 0.002 * (
                1.0
                if epoch < epochs * 0.58
                else 0.35
                if epoch < epochs * 0.84
                else 0.12
            )
            for k in p:
                if wd > 0 and k in ("w", "u", "out"):
                    g[k] += wd * p[k]
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                m_hat = m[k] / (1.0 - 0.9**step)
                v_hat = v[k] / (1.0 - 0.999**step)
                p[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        if (epoch + 1) % 5 == 0 or epoch >= epochs - 10:
            ev_val = t105.evaluate_split(x_val, y_val, p)
            ev_te = t105.evaluate_split(x_te, y_te, p)
            sel_key = (ev_val["top1"], -ev_val["nll"])
            if sel_key > best_key:
                best_key = sel_key
                best_epoch = epoch + 1
                best_p = {k: val.copy() for k, val in p.items()}
            print(
                f"Epoch {epoch + 1:2d}: val={ev_val['top1']:.4f} "
                f"(top3={ev_val['top3']:.4f}, nll={ev_val['nll']:.4f}) | "
                f"test={ev_te['top1']:.4f} (top3={ev_te['top3']:.4f})",
                flush=True,
            )

    assert best_p is not None
    print(f"Selected best validation checkpoint from epoch {best_epoch}", flush=True)
    return best_p, best_epoch


def compute_adaptive_motion_floors(
    train_val_seqs_12fps: list[np.ndarray],
    train_val_labels: list[int],
    idx_to_key: list[str],
) -> dict[str, float]:
    """Compute per-class motion floors blocking static frozen postures across all 116 classes."""
    by_cls: dict[int, list[float]] = {i: [] for i in range(len(idx_to_key))}
    for seq, cls in zip(train_val_seqs_12fps, train_val_labels):
        by_cls[cls].append(t105.motion_v3(seq))

    floors: dict[str, float] = {}
    for cls, vals in by_cls.items():
        if len(vals) < 10:
            continue
        arr = np.asarray(vals, dtype="f")
        min_v = float(arr.min())
        p5_v = float(np.percentile(arr, 5))
        if min_v > 0.05:
            floor = min(0.30, max(0.045, round(min(p5_v * 0.55, min_v * 0.65), 5)))
        elif min_v > 0.012:
            floor = max(0.015, round(min(p5_v * 0.55, min_v * 0.65), 5))
        else:
            floor = 0.012
        floors[idx_to_key[cls]] = floor
    return floors


def calibrate_unified_threshold(ev_val: dict, y_val: np.ndarray) -> tuple[float, list[dict]]:
    """Calibrate confidence threshold on validation split without touching test."""
    preds = ev_val["preds"]
    scores = ev_val["scores"]
    sweep = []
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        mask = scores >= thr
        cov = float(mask.mean())
        acc_on_acc = float((preds[mask] == y_val[mask]).mean()) if mask.any() else 0.0
        sweep.append(
            {
                "threshold": thr,
                "coverage": round(cov, 4),
                "accepted_accuracy": round(acc_on_acc, 4),
            }
        )
    chosen = 0.50
    for entry in sweep:
        if entry["threshold"] <= 0.60 and entry["coverage"] >= 0.94:
            chosen = entry["threshold"]
    return float(chosen), sweep


def summarize_benchmark(
    name: str,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    idx_to_key: list[str],
    params: dict[str, np.ndarray],
    floors: dict[str, float],
    threshold: float,
) -> tuple[dict, np.ndarray, dict]:
    ev = t105.evaluate_split(x_arr, y_arr, params)
    preds = ev["preds"]
    scores = ev["scores"]
    mots = np.array([t105.motion_v3(seq) for seq in x_arr], dtype=np.float64)

    acc = np.zeros(len(y_arr), dtype=bool)
    motion_rejected = 0
    for i in range(len(y_arr)):
        pred_lbl = idx_to_key[int(preds[i])]
        fl = floors.get(pred_lbl)
        if fl is not None and mots[i] < fl:
            motion_rejected += 1
            continue
        if float(scores[i]) >= threshold:
            acc[i] = True

    present = sorted(set(y_arr.tolist()))
    f1s = []
    for c in present:
        tp = int(((preds == c) & (y_arr == c)).sum())
        fp = int(((preds == c) & (y_arr != c)).sum())
        fn = int(((preds != c) & (y_arr == c)).sum())
        f1s.append((2.0 * tp) / max(1, 2 * tp + fp + fn))

    acc_acc = float((preds[acc] == y_arr[acc]).mean()) if acc.any() else 0.0
    stats = {
        "name": name,
        "count": int(len(y_arr)),
        "classes": int(len(present)),
        "top1": round(ev["top1"], 4),
        "top3": round(ev["top3"], 4),
        "top5": round(ev["top5"], 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "nll": round(ev["nll"], 4),
        "accepted_coverage": round(float(acc.mean()), 4),
        "accepted_accuracy": round(acc_acc, 4),
        "motion_rejected_count": int(motion_rejected),
    }
    return ev, acc, stats


def retrain_fresh_hand_v1_fallback(tx_npz: np.lib.npyio.NpzFile, labels: list[str]) -> dict:
    """Retrain pose-free fallback (transaction_signs.json) strictly from fresh Video-extracted hands."""
    raw_all = tx_npz["raw_landmarks"]
    offsets = tx_npz["clip_offsets"]
    cids = tx_npz["class_ids"]
    aspects = tx_npz["aspects"]
    splits = tx_npz["splits"]

    x_tr, y_tr = [], []
    x_te, y_te = [], []
    for k in range(len(cids)):
        seq = raw_all[offsets[k] : offsets[k + 1]]
        aspect = float(aspects[k])
        hands = np.stack([seq[:, 33:54, :2], seq[:, 54:75, :2]], axis=1).copy()
        hands[..., 0] *= aspect
        try:
            feat = ts_v1.features(hands)
        except ValueError:
            continue
        if splits[k] == "train":
            x_tr.append(feat)
            y_tr.append(int(cids[k]))
        elif splits[k] == "test":
            x_te.append(feat)
            y_te.append(int(cids[k]))

    x_tr_arr = np.stack(x_tr, axis=0)
    y_tr_arr = np.array(y_tr, dtype=np.int64)
    x_te_arr = np.stack(x_te, axis=0)
    y_te_arr = np.array(y_te, dtype=np.int64)

    # Train strictly on the train split so val/test stay unseen and static sweep
    # OOD handshapes don't trigger overconfident false positives.
    p = ts_v1.train(x_tr_arr, y_tr_arr, len(labels), seed=37, epochs=55)
    logits = 0.5 * (ts_v1.forward(x_te_arr, p)[0] + ts_v1.forward(ts_v1.reflected(x_te_arr), p)[0])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    scores = probs.max(axis=1)
    acc = scores >= 0.75

    payload = {
        "version": 1,
        "labels": labels,
        "steps": ts_v1.STEPS,
        "features": int(x_tr_arr.shape[-1]),
        "hidden": ts_v1.HIDDEN,
        "threshold": 0.75,
        **{k: np.round(v.ravel(), 5).tolist() for k, v in p.items()},
    }
    TX_FALLBACK_MODEL_PATH.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    return {
        "count": int(len(y_te_arr)),
        "top1": round(float((preds == y_te_arr).mean()), 4),
        "accepted_coverage": round(float(acc.mean()), 4),
        "accepted_accuracy": round(float((preds[acc] == y_te_arr[acc]).mean()), 4),
    }


def _display_label(key: str) -> str:
    if key == "DONT_UNDERSTAND":
        return "DON'T UNDERSTAND"
    if key == "DONT_KNOW":
        return "DON'T KNOW"
    if key == "IM_FINE":
        return "I'M FINE"
    if key == "YOURE_WELCOME":
        return "YOU'RE WELCOME"
    return key.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Only rebuild pose-free v1 transaction_signs.json and update metrics.",
    )
    args = parser.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(2026)

    fsl_labels_info = json.loads(t105.LABELS_PATH.read_text(encoding="utf-8"))
    fsl_keys = [r["key"] for r in fsl_labels_info]
    fsl_set = set(fsl_keys)
    fsl_by_key = {r["key"]: r for r in fsl_labels_info}

    with np.load(TX_CLIPS_PATH, allow_pickle=True) as tx_npz:
        tx_keys = list(tx_npz["labels"])
        tx_raw = tx_npz["raw_landmarks"]
        tx_off = tx_npz["clip_offsets"]
        tx_cids = tx_npz["class_ids"]
        tx_clip_ids = tx_npz["clip_ids"]
        tx_asp = tx_npz["aspects"]
        tx_fps = tx_npz["fps"]
        tx_sp = tx_npz["splits"]
        tx_signer_clusters = tx_npz["signer_clusters"]
        v1_stats = retrain_fresh_hand_v1_fallback(tx_npz, tx_keys)

    print(
        f"Refreshed pose-free v1 fallback from fresh hands -> {TX_FALLBACK_MODEL_PATH.name}: {v1_stats}",
        flush=True,
    )
    if args.fallback_only:
        if METRICS_OUT_PATH.exists():
            metrics = json.loads(METRICS_OUT_PATH.read_text(encoding="utf-8"))
            metrics["pose_free_v1_fallback"] = v1_stats
            METRICS_OUT_PATH.write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
        return

    new_keys = sorted(k for k in tx_keys if k not in fsl_set)
    idx_to_key = fsl_keys + new_keys
    key_to_idx = {k: i for i, k in enumerate(idx_to_key)}
    num_classes = len(idx_to_key)
    assert num_classes == 116, f"Expected 116 unified classes, got {num_classes}"

    x_tr, y_tr = [], []
    x_val_fsl, y_val_fsl = [], []
    x_te_fsl, y_te_fsl = [], []
    x_val_tx, y_val_tx = [], []
    x_te_tx, y_te_tx = [], []
    tx_te_meta: list[dict] = []
    tv_12fps_seqs, tv_12fps_labels = [], []

    # 1. FSL-105 clips
    with np.load(t105.CLIPS_PATH) as d:
        raw_all = d["raw_landmarks"]
        offsets = d["clip_offsets"]
        m_indices = d["model_indices"]
        aspects = d["aspects"]
        fps_arr = d["fps"]
        splits = d["splits"]

    for k in range(len(m_indices)):
        raw = raw_all[offsets[k] : offsets[k + 1]]
        idx = int(m_indices[k])
        aspect = float(aspects[k])
        fps = float(fps_arr[k])
        sp = str(splits[k])

        if sp == "train":
            stride_12 = max(1, round(fps / 12.0))
            base_12 = t105.encode_v4(raw[::stride_12], aspect)
            if base_12 is not None:
                tv_12fps_seqs.append(base_12)
                tv_12fps_labels.append(idx)

            for rate in (18.0, 12.0, 9.0, 6.0, 4.5):
                stride = max(1, round(fps / rate))
                sub = raw[int(rng.integers(stride)) :: stride]
                if rate in (12.0, 6.0):
                    sub = t105.augment_raw(sub, aspect, rng, trim=(rate == 6.0))
                if rate == 9.0 and len(sub) > 6:
                    keep = rng.random(len(sub)) > 0.15
                    if int(keep.sum()) >= 3:
                        sub = sub[keep]
                feat = t105.encode_v4(sub, aspect)
                if feat is not None:
                    x_tr.append(feat)
                    y_tr.append(idx)

            for extra_rate, do_trim in ((12.0, True), (9.0, False), (15.0, True)):
                stride = max(1, round(fps / extra_rate))
                sub = t105.augment_raw(
                    raw[int(rng.integers(stride)) :: stride], aspect, rng, trim=do_trim
                )
                feat = t105.encode_v4(sub, aspect)
                if feat is not None:
                    x_tr.append(feat)
                    y_tr.append(idx)
        else:
            stride = max(1, round(fps / 12.0))
            feat = t105.encode_v4(raw[::stride], aspect)
            if feat is None:
                continue
            if sp == "val":
                x_val_fsl.append(feat)
                y_val_fsl.append(idx)
                tv_12fps_seqs.append(feat)
                tv_12fps_labels.append(idx)
            elif sp == "test":
                x_te_fsl.append(feat)
                y_te_fsl.append(idx)

    # 2. Greeting keypoints extra (same exact filter as train_fsl105.py)
    if t105.GREETING_KEYPOINTS_DIR.is_dir():
        for gdir in sorted(t105.GREETING_KEYPOINTS_DIR.iterdir()):
            if not gdir.is_dir() or gdir.name not in key_to_idx:
                continue
            idx = key_to_idx[gdir.name]
            for npz_path in sorted(gdir.glob("*.npz"), key=lambda p: int(p.stem)):
                clip_num = int(npz_path.stem)
                if clip_num >= 15 or clip_num % 5 == 4:
                    continue
                with np.load(npz_path) as gd:
                    g_raw = gd["raw_landmarks"]
                    g_aspect = float(gd["aspect"])
                    g_fps = float(gd["fps"])
                for rate in (12.0, 8.0, 5.0):
                    stride = max(1, round(g_fps / rate))
                    feat = t105.encode_v4(
                        g_raw[int(rng.integers(stride)) :: stride], g_aspect
                    )
                    if feat is not None:
                        x_tr.append(feat)
                        y_tr.append(idx)

    # 3. Transaction dataset (freshly extracted from coordinates/videos)
    for k in range(len(tx_cids)):
        key = tx_keys[int(tx_cids[k])]
        idx = key_to_idx[key]
        raw = tx_raw[tx_off[k] : tx_off[k + 1]]
        aspect = float(tx_asp[k])
        fps = float(tx_fps[k])
        sp = str(tx_sp[k])
        is_new = key not in fsl_set

        if sp == "train":
            stride_12 = max(1, round(fps / 12.0))
            base_12 = t105.encode_v4(raw[::stride_12], aspect)
            if base_12 is not None:
                tv_12fps_seqs.append(base_12)
                tv_12fps_labels.append(idx)

            configs = [
                (18.0, False, False),
                (12.0, True, False),
                (9.0, False, True),
                (6.0, True, True),
            ]
            if is_new:
                configs.append((15.0, True, True))
            for rate, do_aug, do_trim in configs:
                stride = max(1, round(fps / rate))
                sub = raw[int(rng.integers(stride)) :: stride]
                if do_aug:
                    sub = t105.augment_raw(sub, aspect, rng, trim=do_trim)
                elif rate == 9.0 and len(sub) > 6:
                    keep = rng.random(len(sub)) > 0.15
                    if int(keep.sum()) >= 3:
                        sub = sub[keep]
                feat = t105.encode_v4(sub, aspect)
                if feat is not None:
                    x_tr.append(feat)
                    y_tr.append(idx)
        else:
            stride = max(1, round(fps / 12.0))
            feat = t105.encode_v4(raw[::stride], aspect)
            if feat is None:
                continue
            if sp == "val":
                x_val_tx.append(feat)
                y_val_tx.append(idx)
                tv_12fps_seqs.append(feat)
                tv_12fps_labels.append(idx)
            elif sp == "test":
                x_te_tx.append(feat)
                y_te_tx.append(idx)
                tx_te_meta.append(
                    {
                        "tx_npz_idx": k,
                        "key": key,
                        "clip_id": int(tx_clip_ids[k]),
                        "signer_cluster": int(tx_signer_clusters[k]),
                        "model_idx": idx,
                        "aspect": aspect,
                        "fps": fps,
                        "stride": stride,
                    }
                )

    x_tr_arr = np.asarray(x_tr, dtype="f")
    y_tr_arr = np.asarray(y_tr, dtype=np.int64)
    x_val_fsl_arr = np.asarray(x_val_fsl, dtype="f")
    y_val_fsl_arr = np.asarray(y_val_fsl, dtype=np.int64)
    x_te_fsl_arr = np.asarray(x_te_fsl, dtype="f")
    y_te_fsl_arr = np.asarray(y_te_fsl, dtype=np.int64)
    x_val_tx_arr = np.asarray(x_val_tx, dtype="f")
    y_val_tx_arr = np.asarray(y_val_tx, dtype=np.int64)
    x_te_tx_arr = np.asarray(x_te_tx, dtype="f")
    y_te_tx_arr = np.asarray(y_te_tx, dtype=np.int64)

    x_val_all = np.concatenate([x_val_fsl_arr, x_val_tx_arr], axis=0)
    y_val_all = np.concatenate([y_val_fsl_arr, y_val_tx_arr], axis=0)
    x_te_all = np.concatenate([x_te_fsl_arr, x_te_tx_arr], axis=0)
    y_te_all = np.concatenate([y_te_fsl_arr, y_te_tx_arr], axis=0)

    print(
        f"Encoded Unified 116-Class in {time.time() - t0:.1f}s: "
        f"train={x_tr_arr.shape}, val={x_val_all.shape} "
        f"(fsl={len(x_val_fsl_arr)}, tx={len(x_val_tx_arr)}), "
        f"test={x_te_all.shape} (fsl={len(x_te_fsl_arr)}, tx={len(x_te_tx_arr)})",
        flush=True,
    )

    best_p, best_epoch = train_gru_unified(
        x_tr_arr,
        y_tr_arr,
        x_val_all,
        y_val_all,
        x_te_all,
        y_te_all,
        num_classes,
        epochs=75,
        wd=2.5e-4,
        feat_drop=0.12,
        smooth=0.04,
    )

    ev_val_all = t105.evaluate_split(x_val_all, y_val_all, best_p)
    motion_floors = compute_adaptive_motion_floors(
        tv_12fps_seqs, tv_12fps_labels, idx_to_key
    )
    threshold, val_sweep = calibrate_unified_threshold(ev_val_all, y_val_all)

    ev_fsl, acc_fsl, st_fsl = summarize_benchmark(
        "FSL-105 Signer-4 Test (104 classes)",
        x_te_fsl_arr,
        y_te_fsl_arr,
        idx_to_key,
        best_p,
        motion_floors,
        threshold,
    )
    ev_tx, acc_tx, st_tx = summarize_benchmark(
        "Transaction Unseen-Signer Test (All 26 classes)",
        x_te_tx_arr,
        y_te_tx_arr,
        idx_to_key,
        best_p,
        motion_floors,
        threshold,
    )
    is_new_mask = np.array([idx_to_key[y] not in fsl_set for y in y_te_tx_arr])
    _, _, st_tx_new = summarize_benchmark(
        "Transaction New-12 Unseen Test",
        x_te_tx_arr[is_new_mask],
        y_te_tx_arr[is_new_mask],
        idx_to_key,
        best_p,
        motion_floors,
        threshold,
    )
    _, _, st_tx_ovl = summarize_benchmark(
        "Transaction Overlap-14 Unseen Test",
        x_te_tx_arr[~is_new_mask],
        y_te_tx_arr[~is_new_mask],
        idx_to_key,
        best_p,
        motion_floors,
        threshold,
    )
    _, _, st_combined = summarize_benchmark(
        "Combined Held-out Test (All 116 classes)",
        x_te_all,
        y_te_all,
        idx_to_key,
        best_p,
        motion_floors,
        threshold,
    )

    # Frozen static posture false-trigger check
    frozen_all = np.concatenate(
        [np.repeat(seq[8:9], t105.STEPS, axis=0)[None, ...] for seq in x_te_all],
        axis=0,
    )
    frozen_all[:, :, t105.STATIC_FEATURES :] = 0.0
    ev_frozen = t105.evaluate_split(frozen_all, y_te_all, best_p)
    frozen_fa = 0
    for i in range(len(y_te_all)):
        pred_lbl = idx_to_key[int(ev_frozen["preds"][i])]
        fl = motion_floors.get(pred_lbl)
        if fl is not None and t105.motion_v3(frozen_all[i]) < fl:
            continue
        if float(ev_frozen["scores"][i]) >= threshold:
            frozen_fa += 1

    # Mirror invariance check
    _, p_dir = t105.predict_probs(x_te_all, best_p)
    _, p_mir = t105.predict_probs(t105.mirror_v3(x_te_all), best_p)
    mirror_max_diff = float(np.abs(p_dir - p_mir).max())

    # Write combined model JSON & labels metadata
    model_payload = {
        "version": t105.VERSION,
        "steps": t105.STEPS,
        "features": t105.FEATURES,
        "hidden": t105.HIDDEN,
        "labels": idx_to_key,
        "threshold": round(threshold, 4),
        "motionFloors": motion_floors,
        "w": np.round(best_p["w"].ravel(), 6).tolist(),
        "u": np.round(best_p["u"].ravel(), 6).tolist(),
        "b": np.round(best_p["b"].ravel(), 6).tolist(),
        "out": np.round(best_p["out"].ravel(), 6).tolist(),
        "bias": np.round(best_p["bias"].ravel(), 6).tolist(),
    }
    COMBINED_MODEL_PATH.write_text(
        json.dumps(model_payload, separators=(",", ":")), encoding="utf-8"
    )

    tx_set = set(tx_keys)
    labels_meta = []
    for idx, key in enumerate(idx_to_key):
        in_fsl = key in fsl_set
        in_tx = key in tx_set
        if in_fsl and in_tx:
            source = "fsl105+transaction"
        elif in_fsl:
            source = "fsl105"
        else:
            source = "transaction"
        fsl_row = fsl_by_key.get(key)
        labels_meta.append(
            {
                "model_index": idx,
                "id": fsl_row["id"] if fsl_row is not None else 1000 + idx,
                "key": key,
                "label": fsl_row["label"] if fsl_row is not None else _display_label(key),
                "category": fsl_row["category"] if fsl_row is not None else "TRANSACTION",
                "source": source,
            }
        )
    COMBINED_LABELS_PATH.write_text(
        json.dumps(labels_meta, indent=2), encoding="utf-8"
    )

    # Export TX 26-class test_samples.json (one unseen-signer clip per TX class) in exact Dart schema
    by_tx_cls: dict[str, list[int]] = {}
    for pos, meta in enumerate(tx_te_meta):
        by_tx_cls.setdefault(meta["key"], []).append(pos)

    samples_out = []
    for key in tx_keys:
        positions = by_tx_cls[key]
        correct_pos = [
            p for p in positions if int(ev_tx["preds"][p]) == key_to_idx[key]
        ]
        pick = (
            max(correct_pos, key=lambda p: float(ev_tx["scores"][p]))
            if correct_pos
            else positions[0]
        )
        meta = tx_te_meta[pick]
        raw = tx_raw[tx_off[meta["tx_npz_idx"]] : tx_off[meta["tx_npz_idx"] + 1]]
        strided = raw[:: meta["stride"]]
        frames_out = []
        for frame in strided:
            pose5 = [
                [round(float(frame[p_idx, 0]), 6), round(float(frame[p_idx, 1]), 6)]
                for p_idx in t105.POSE_ANCHORS
            ]
            h0 = [
                [
                    round(float(frame[33 + j, 0]), 6),
                    round(float(frame[33 + j, 1]), 6),
                    round(float(frame[33 + j, 2]), 6),
                ]
                for j in range(21)
            ]
            h1 = (
                [
                    [
                        round(float(frame[54 + j, 0]), 6),
                        round(float(frame[54 + j, 1]), 6),
                        round(float(frame[54 + j, 2]), 6),
                    ]
                    for j in range(21)
                ]
                if np.any(frame[54:75] != 0)
                else None
            )
            frames_out.append({"pose": pose5, "hand0": h0, "hand1": h1})
        top3_idx = np.argsort(-ev_tx["probs"][pick])[:3]
        samples_out.append(
            {
                "label": key,
                "model_index": meta["model_idx"],
                "clip_id": meta["clip_id"],
                "signer_cluster": meta["signer_cluster"],
                "in_fsl105": key in fsl_set,
                "aspect": round(float(meta["aspect"]), 6),
                "fps": round(float(meta["fps"]), 3),
                "expected_pred": idx_to_key[int(ev_tx["preds"][pick])],
                "expected_score": round(float(ev_tx["scores"][pick]), 6),
                "expected_motion": round(t105.motion_v3(x_te_tx_arr[pick]), 6),
                "top3": [
                    [idx_to_key[int(ti)], round(float(ev_tx["probs"][pick, ti]), 6)]
                    for ti in top3_idx
                ],
                "frames": frames_out,
            }
        )
    SAMPLES_OUT_PATH.write_text(
        json.dumps(samples_out, separators=(",", ":")), encoding="utf-8"
    )

    # Confusion matrix across TX 26 unseen test classes (columns include all 116 or predicted classes)
    with CONFUSION_OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred"] + idx_to_key)
        for key in tx_keys:
            cid = key_to_idx[key]
            m = y_te_tx_arr == cid
            counts = np.bincount(ev_tx["preds"][m], minlength=num_classes)
            writer.writerow([key] + counts.tolist())

    # Per-class TX breakdown
    tx_per_class = []
    for key in tx_keys:
        cid = key_to_idx[key]
        m = y_te_tx_arr == cid
        tp = int((ev_tx["preds"][m] == cid).sum())
        ac = int(acc_tx[m].sum())
        ac_ok = int((acc_tx[m] & (ev_tx["preds"][m] == cid)).sum())
        tx_per_class.append(
            {
                "key": key,
                "in_fsl105": key in fsl_set,
                "test_count": int(m.sum()),
                "top1_correct": tp,
                "top1": round(tp / max(1, int(m.sum())), 4),
                "accepted_count": ac,
                "accepted_correct": ac_ok,
                "wrong_preds": [
                    idx_to_key[int(p)] for p in ev_tx["preds"][m] if p != cid
                ],
            }
        )

    metrics = {
        "model": {
            "version": t105.VERSION,
            "steps": t105.STEPS,
            "features": t105.FEATURES,
            "hidden": t105.HIDDEN,
            "num_classes": num_classes,
            "fsl105_classes": len(fsl_keys),
            "new_tx_classes": len(new_keys),
            "shared_classes": len(fsl_set & tx_set),
            "best_epoch": best_epoch,
            "threshold": round(threshold, 4),
            "motion_floors_count": len(motion_floors),
        },
        "validation": {
            "count": int(len(y_val_all)),
            "top1": round(ev_val_all["top1"], 4),
            "top3": round(ev_val_all["top3"], 4),
            "nll": round(ev_val_all["nll"], 4),
            "threshold_sweep": val_sweep,
        },
        "benchmarks": {
            "combined_test": st_combined,
            "fsl105_signer4_test": st_fsl,
            "tx_all26_unseen_test": st_tx,
            "tx_new12_unseen_test": st_tx_new,
            "tx_overlap14_unseen_test": st_tx_ovl,
        },
        "safety": {
            "frozen_static_test_clips": int(len(y_te_all)),
            "frozen_static_false_accepts": int(frozen_fa),
            "mirror_max_prob_diff": round(mirror_max_diff, 8),
        },
        "pose_free_v1_fallback": v1_stats,
        "tx_per_class": tx_per_class,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    METRICS_OUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"\n=== Unified 116-Class Done ({time.time() - t0:.1f}s, epoch={best_epoch}, thr={threshold}) ===",
        flush=True,
    )
    for b in metrics["benchmarks"].values():
        print(
            f"  {b['name']:46s}: top1={b['top1']:.4f} top3={b['top3']:.4f} "
            f"macroF1={b['macro_f1']:.4f} cov={b['accepted_coverage']:.4f} acc={b['accepted_accuracy']:.4f}",
            flush=True,
        )
    print(
        f"  Frozen FA={frozen_fa}/{len(y_te_all)} | Mirror max diff={mirror_max_diff:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
