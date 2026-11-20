"""Train and evaluate the Visualin v4 dynamic FSL-105 classifier.

Reads extracted keypoints from ``fsl105_keypoints/fsl105_clips.npz`` and
``assets/ml/fsl105_labels.json`` (104 active classes, excluding ID 2 GOOD EVENING),
trains a 322-feature 16-step bidirectional-mirror-invariant GRU (hidden=128) with
clean main-island filtering, duplicate-hand suppression, palm orientation & wrist
rotation, forearm/upper-arm kinematics, head-height vertical normalization,
finger curls, thumb-tip separation, MCP flexion, and 2-hand coordination cues,
calibrates conservative ``motionFloors`` and softmax confidence ``threshold``
from validation/train+val statistics (never leaking Signer 4 ``test``), evaluates
held-out Signer 4 generalization, and exports:
  - ``assets/ml/fsl105_signs.json``
  - ``fsl105_keypoints/evaluation_metrics.json``
  - ``fsl105_keypoints/confusion_matrix.csv``
  - ``fsl105_keypoints/test_samples.json``
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLIPS_PATH = ROOT / "fsl105_keypoints" / "fsl105_clips.npz"
LABELS_PATH = ROOT / "assets" / "ml" / "fsl105_labels.json"
GREETING_KEYPOINTS_DIR = ROOT / "greeting_keypoints"
MODEL_OUT_PATH = ROOT / "assets" / "ml" / "fsl105_signs.json"
METRICS_OUT_PATH = ROOT / "fsl105_keypoints" / "evaluation_metrics.json"
CONFUSION_OUT_PATH = ROOT / "fsl105_keypoints" / "confusion_matrix.csv"
SAMPLES_OUT_PATH = ROOT / "fsl105_keypoints" / "test_samples.json"

VERSION = 4
STEPS = 16
HIDDEN = 128
STATIC_FEATURES = 213
VEL_INDICES = (
    list(range(84, 88))
    + list(range(91, 111))
    + list(range(122, 142))
    + [8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 50, 51, 58, 59, 66, 67, 74, 75, 82, 83]
    + list(range(142, 156))
    + list(range(166, 194))
    + list(range(210, 213))
)
FEATURES = STATIC_FEATURES + len(VEL_INDICES)  # 213 + 109 = 322
STATIC_X_FLIP = set(
    list(range(0, 88, 2))
    + list(range(91, 111, 2))
    + [140, 166, 168, 170, 171, 173, 175, 176, 179, 182, 185, 188, 192]
)
VEL_X_FLIP = [
    STATIC_FEATURES + k for k, src in enumerate(VEL_INDICES) if src in STATIC_X_FLIP
]
X_FLIP = sorted(STATIC_X_FLIP | set(VEL_X_FLIP))
POSE_ANCHORS = [0, 11, 12, 13, 14]
PALM_ANCHORS = [5, 9, 13, 17]
CURL_CHAINS = (
    (1, 2, 2, 4),
    (5, 6, 6, 8),
    (9, 10, 10, 12),
    (13, 14, 14, 16),
    (17, 18, 18, 20),
)
MCP_CHAINS = (
    (0, 1, 1, 2),
    (0, 5, 5, 6),
    (0, 9, 9, 10),
    (0, 13, 13, 14),
    (0, 17, 17, 18),
)


def _palm_size(hand_xy: np.ndarray) -> float:
    return float(
        np.mean(np.linalg.norm(hand_xy[PALM_ANCHORS] - hand_xy[0:1], axis=-1))
    )


def filter_main_islands(keep_mask: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Drop tiny stray detection bursts separated from the main sign by large empty gaps."""
    idx = np.where(keep_mask)[0]
    if len(idx) == 0:
        return idx
    cuts = np.where(np.diff(idx) >= max_gap)[0] + 1
    if len(cuts) == 0:
        return idx
    islands = np.split(idx, cuts)
    best = max(len(isl) for isl in islands)
    kept = [isl for isl in islands if len(isl) >= max(8, int(round(0.35 * best)))]
    if not kept:
        kept = [islands[int(np.argmax([len(isl) for isl in islands]))]]
    return np.concatenate(kept)


def stabilize_slots(
    hands: np.ndarray, visible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match SignClassifier._predictV4 slot ordering, duplicate suppression, and crossing hysteresis."""
    hands = hands.copy()
    visible = visible.copy()
    xy = hands[..., :2]
    order = np.argsort(
        np.where(visible, xy[:, :, 0, 1], np.inf), axis=1, kind="stable"
    )
    hands = np.take_along_axis(hands, order[:, :, None, None], axis=1)
    visible = np.take_along_axis(visible, order, axis=1)
    for i in range(len(hands)):
        if visible[i, 0] and visible[i, 1]:
            palm = max(
                _palm_size(hands[i, 0, :, :2]),
                _palm_size(hands[i, 1, :, :2]),
                1e-4,
            )
            mean_d = float(
                np.linalg.norm(
                    hands[i, 0, :, :2] - hands[i, 1, :, :2], axis=-1
                ).mean()
            )
            if mean_d < 0.45 * palm:
                hands[i, 1] = 0.0
                visible[i, 1] = False
    for i in range(1, len(hands)):
        if visible[i, 0] and visible[i, 1] and visible[i - 1, 0] and visible[i - 1, 1]:
            w0, w1 = hands[i, 0, 0, :2], hands[i, 1, 0, :2]
            pw0, pw1 = hands[i - 1, 0, 0, :2], hands[i - 1, 1, 0, :2]
            palm = max(_palm_size(hands[i, 0, :, :2]), 0.02)
            if abs(w0[1] - w1[1]) < 1.25 * palm:
                d_keep = float(np.sum((w0 - pw0) ** 2) + np.sum((w1 - pw1) ** 2))
                d_swap = float(np.sum((w1 - pw0) ** 2) + np.sum((w0 - pw1) ** 2))
                if d_swap + 1e-6 < d_keep:
                    hands[i] = hands[i, ::-1].copy()
    return hands, visible


def resolve_pose(pose: np.ndarray) -> np.ndarray | None:
    """Carry forward valid 5-point upper-body pose and backfill leading gaps."""
    width = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=1)
    anchors = pose[:, POSE_ANCHORS]
    valid = (
        (width > 0.025)
        & np.isfinite(anchors).all(axis=(1, 2))
        & np.any(anchors != 0, axis=2).all(axis=1)
    )
    if not valid.any():
        return None
    out = pose.copy()
    last = None
    first_idx = None
    for i in range(len(out)):
        if valid[i]:
            last = out[i]
            if first_idx is None:
                first_idx = i
        elif last is not None:
            out[i] = last
    for i in range(first_idx):
        out[i] = out[first_idx]
    return out


def encode_v4(raw: np.ndarray, aspect: float) -> np.ndarray | None:
    """Encode raw (T, 75, 3) landmarks into a (16, 322) v4 feature sequence."""
    work = raw.astype("f", copy=True)
    work[..., 0] *= aspect
    raw_hands = work[:, 33:].reshape(-1, 2, 21, 3)
    xy_all = raw_hands[..., :2]
    palms_raw = np.mean(
        np.linalg.norm(
            xy_all[:, :, PALM_ANCHORS] - xy_all[:, :, 0:1], axis=-1
        ),
        axis=2,
    )
    finite_mask = np.isfinite(xy_all).all(axis=(2, 3))
    nonzero_mask = np.any(xy_all != 0, axis=(2, 3))
    valid_hand = finite_mask & nonzero_mask & (palms_raw > 1e-4)
    keep_idx = filter_main_islands(valid_hand.any(axis=1), max_gap=10)
    if len(keep_idx) < 3:
        return None

    filtered_hands = raw_hands[keep_idx].copy()
    filtered_vis = valid_hand[keep_idx].copy()
    # Zero invalid slots so sorting places valid hand first.
    filtered_hands[~filtered_vis] = 0.0

    hands, visible = stabilize_slots(filtered_hands, filtered_vis)
    if int(visible[:, 0].sum()) == 0:
        return None
    pose = resolve_pose(work[keep_idx, :33, :2])
    if pose is None:
        return None

    xy = hands[..., :2]
    n_frames = len(hands)
    wrist = xy[:, :, 0, :]
    local = xy - wrist[:, :, None, :]
    palm = np.mean(np.linalg.norm(local[:, :, PALM_ANCHORS], axis=-1), axis=2)
    palm_safe = np.maximum(palm[:, :, None, None], 1e-4)
    v0_idx = np.where(visible[:, 0])[0]
    primary_scale = max(float(np.median(palm[v0_idx, 0])), 1e-4)
    norm_xy = np.where(visible[:, :, None, None], local / palm_safe, 0.0)
    shape = norm_xy.reshape(n_frames, 84)

    tips_local = norm_xy[:, :, [4, 8, 12, 16, 20]]
    openness = np.linalg.norm(tips_local, axis=-1).reshape(n_frames, 10)
    pinch = np.linalg.norm(
        tips_local[:, :, 1:] - tips_local[:, :, :1], axis=-1
    ).reshape(n_frames, 8)
    openness = np.where(
        np.repeat(visible, 5, axis=1), openness, 0.0
    )
    pinch = np.where(
        np.repeat(visible, 4, axis=1), pinch, 0.0
    )

    anchor_w0 = wrist[v0_idx[0], 0]
    v0_col = visible[:, 0:1].astype("f")
    v1_col = visible[:, 1:2].astype("f")
    both_col = (visible[:, 0] & visible[:, 1])[:, None].astype("f")
    movement0 = ((wrist[:, 0] - anchor_w0) / primary_scale) * v0_col
    separation = ((wrist[:, 1] - wrist[:, 0]) / primary_scale) * both_col
    idx_sep = ((xy[:, 1, 8] - xy[:, 0, 8]) / primary_scale) * both_col

    center = pose[:, [11, 12]].mean(axis=1)
    w_scale = np.maximum(
        np.linalg.norm(pose[:, 11] - pose[:, 12], axis=1)[:, None], 0.025
    )
    head_h = np.maximum(
        np.linalg.norm(center - pose[:, 0], axis=1)[:, None], 0.25 * w_scale
    )

    body = np.zeros((n_frames, 21), dtype="f")
    depth = np.zeros((n_frames, 10), dtype="f")
    adj_spread = np.zeros((n_frames, 6), dtype="f")
    thumb_pip = np.zeros((n_frames, 8), dtype="f")
    curls = np.zeros((n_frames, 10), dtype="f")
    palm_ori = np.zeros((n_frames, 10), dtype="f")
    arm_ori = np.zeros((n_frames, 12), dtype="f")
    opp_elbows = np.zeros((n_frames, 2), dtype="f")
    thumb_tips = np.zeros((n_frames, 6), dtype="f")
    mcp_curls = np.zeros((n_frames, 10), dtype="f")

    for slot in range(2):
        vis_s = visible[:, slot : slot + 1]
        w_pt = xy[:, slot, 0]
        idx_pt = xy[:, slot, 8]
        thm_pt = xy[:, slot, 4]
        elbows = pose[:, [13, 14]]
        shoulders = pose[:, [11, 12]]
        d_el = np.linalg.norm(elbows - w_pt[:, None], axis=2)
        near = np.argmin(d_el, axis=1)
        elbow = elbows[np.arange(n_frames), near]
        shoulder = shoulders[np.arange(n_frames), near]
        if slot == 0:
            opp_elbow = elbows[np.arange(n_frames), 1 - near]
            opp_elbows = np.where(vis_s, (w_pt - opp_elbow) / w_scale, 0.0)

        offsets = np.concatenate(
            [
                w_pt - center,
                elbow - center,
                w_pt - pose[:, 0],
                idx_pt - pose[:, 0],
                thm_pt - pose[:, 0],
            ],
            axis=1,
        )
        body[:, slot * 10 : (slot + 1) * 10] = np.where(
            vis_s, offsets / w_scale, 0.0
        )
        raw_depth = (
            (hands[:, slot, [4, 8, 12, 16, 20], 2] - hands[:, slot, 0:1, 2])
            * aspect
            / np.maximum(palm[:, slot : slot + 1], 1e-4)
        )
        depth[:, slot * 5 : (slot + 1) * 5] = np.where(
            vis_s, raw_depth, 0.0
        )

        s_xy = norm_xy[:, slot]
        sp = np.stack(
            [
                np.linalg.norm(s_xy[:, 8] - s_xy[:, 12], axis=1),
                np.linalg.norm(s_xy[:, 12] - s_xy[:, 16], axis=1),
                np.linalg.norm(s_xy[:, 16] - s_xy[:, 20], axis=1),
            ],
            axis=1,
        )
        adj_spread[:, slot * 3 : (slot + 1) * 3] = np.where(vis_s, sp, 0.0)

        tp = np.stack(
            [
                np.linalg.norm(s_xy[:, 4] - s_xy[:, 5], axis=1),
                np.linalg.norm(s_xy[:, 4] - s_xy[:, 6], axis=1),
                np.linalg.norm(s_xy[:, 4] - s_xy[:, 10], axis=1),
                np.linalg.norm(s_xy[:, 4] - s_xy[:, 14], axis=1),
            ],
            axis=1,
        )
        thumb_pip[:, slot * 4 : (slot + 1) * 4] = np.where(vis_s, tp, 0.0)

        tt = np.stack(
            [
                np.linalg.norm(s_xy[:, 4] - s_xy[:, 12], axis=1),
                np.linalg.norm(s_xy[:, 4] - s_xy[:, 16], axis=1),
                np.linalg.norm(s_xy[:, 4] - s_xy[:, 20], axis=1),
            ],
            axis=1,
        )
        thumb_tips[:, slot * 3 : (slot + 1) * 3] = np.where(vis_s, tt, 0.0)

        c_list = []
        for a, b, c, d_i in CURL_CHAINS:
            u = s_xy[:, b] - s_xy[:, a]
            v = s_xy[:, d_i] - s_xy[:, c]
            denom = np.maximum(
                np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-6
            )
            c_list.append(np.sum(u * v, axis=1) / denom)
        curls[:, slot * 5 : (slot + 1) * 5] = np.where(
            vis_s, np.stack(c_list, axis=1), 0.0
        )

        m_list = []
        for a, b, c, d_i in MCP_CHAINS:
            u = s_xy[:, b] - s_xy[:, a]
            v = s_xy[:, d_i] - s_xy[:, c]
            denom = np.maximum(
                np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-6
            )
            m_list.append(np.sum(u * v, axis=1) / denom)
        mcp_curls[:, slot * 5 : (slot + 1) * 5] = np.where(
            vis_s, np.stack(m_list, axis=1), 0.0
        )

        u_p = s_xy[:, 9] - s_xy[:, 0]
        u_p /= np.maximum(np.linalg.norm(u_p, axis=1, keepdims=True), 1e-6)
        v_k = s_xy[:, 17] - s_xy[:, 5]
        v_k /= np.maximum(np.linalg.norm(v_k, axis=1, keepdims=True), 1e-6)
        cross = (u_p[:, 0] * v_k[:, 1] - u_p[:, 1] * v_k[:, 0])[:, None]
        palm_ori[:, slot * 5 : (slot + 1) * 5] = np.where(
            vis_s, np.concatenate([u_p, v_k, cross], axis=1), 0.0
        )

        fa = w_pt - elbow
        fa_norm = np.maximum(np.linalg.norm(fa, axis=1, keepdims=True), 0.02)
        fa_unit = fa / fa_norm
        fa_len = np.linalg.norm(fa, axis=1, keepdims=True) / w_scale
        ua = elbow - shoulder
        ua_norm = np.maximum(np.linalg.norm(ua, axis=1, keepdims=True), 0.02)
        ua_unit = ua / ua_norm
        wy_head = (w_pt[:, 1:2] - pose[:, 0, 1:2]) / head_h
        arm_ori[:, slot * 6 : (slot + 1) * 6] = np.where(
            vis_s,
            np.concatenate([fa_unit, fa_len, ua_unit, wy_head], axis=1),
            0.0,
        )

    body[:, 20] = 1.0
    mean_w0 = wrist[v0_idx, 0].mean(axis=0, keepdims=True)
    centered_w0 = ((wrist[:, 0] - mean_w0) / w_scale) * v0_col
    head_tips0 = np.concatenate(
        [
            ((xy[:, 0, 8, 1:2] - pose[:, 0, 1:2]) / head_h) * v0_col,
            ((xy[:, 0, 4, 1:2] - pose[:, 0, 1:2]) / head_h) * v0_col,
        ],
        axis=1,
    )
    extra_coord = np.concatenate([centered_w0, head_tips0, opp_elbows], axis=1)

    palm_dot = (
        palm_ori[:, 0:1] * palm_ori[:, 5:6] + palm_ori[:, 1:2] * palm_ori[:, 6:7]
    ) * both_col
    palm_dist = (
        np.linalg.norm(xy[:, 0, 9] - xy[:, 1, 9], axis=1, keepdims=True)
        / primary_scale
    ) * both_col
    tip_to_opp_wrist = (
        np.linalg.norm(xy[:, 0, 8] - xy[:, 1, 0], axis=1, keepdims=True)
        / primary_scale
    ) * both_col
    inter_coord = np.concatenate([palm_dot, palm_dist, tip_to_opp_wrist], axis=1)

    static_feat = np.concatenate(
        [
            shape,
            movement0,
            separation,
            v1_col,
            np.linalg.norm(movement0, axis=1, keepdims=True),
            np.linalg.norm(separation, axis=1, keepdims=True),
            body,
            depth,
            openness,
            pinch,
            idx_sep,
            adj_spread,
            thumb_pip,
            curls,
            palm_ori,
            arm_ori,
            extra_coord,
            thumb_tips,
            mcp_curls,
            inter_coord,
        ],
        axis=1,
    )
    assert static_feat.shape[1] == STATIC_FEATURES, static_feat.shape
    static_feat = np.clip(static_feat, -5.0, 5.0)

    t = np.linspace(0, n_frames - 1, STEPS)
    lo = np.floor(t).astype(int)
    hi = np.ceil(t).astype(int)
    frac = (t - lo)[:, None]
    seq = (static_feat[lo] + (static_feat[hi] - static_feat[lo]) * frac).astype("f")

    vel = np.zeros((STEPS, len(VEL_INDICES)), dtype="f")
    vel[1:] = np.clip(
        (seq[1:, VEL_INDICES] - seq[:-1, VEL_INDICES]) * 3.0, -5.0, 5.0
    )
    return np.concatenate([seq, vel], axis=1)


def augment_raw(
    raw: np.ndarray, aspect: float, rng: np.random.Generator, trim: bool = False
) -> np.ndarray:
    """Natural signing style, speed, body-proportion, locus, occlusion, and depth augmentation."""
    out = raw.astype("f", copy=True)

    if len(out) >= 10 and rng.random() < 0.65:
        gamma = float(rng.uniform(0.78, 1.28))
        n = len(out)
        u = np.linspace(0.0, 1.0, n) ** gamma
        idx = np.clip(np.round(u * (n - 1)).astype(int), 0, n - 1)
        out = out[idx]

    if trim and len(out) >= 12:
        left_trim = int(rng.integers(0, min(4, len(out) // 5 + 1)))
        right_trim = int(rng.integers(0, min(4, len(out) // 5 + 1)))
        out = out[left_trim : len(out) - right_trim if right_trim > 0 else len(out)]
    elif not trim and len(out) >= 8 and rng.random() < 0.30:
        pad_start = int(rng.integers(0, 3))
        pad_end = int(rng.integers(0, 3))
        parts = []
        if pad_start > 0:
            parts.append(np.repeat(out[:1], pad_start, axis=0))
        parts.append(out)
        if pad_end > 0:
            parts.append(np.repeat(out[-1:], pad_end, axis=0))
        out = np.concatenate(parts, axis=0)

    out[..., 0] *= aspect
    pres_pose = np.any(out[:, :33, :2] != 0, axis=2)
    pres_hand = np.any(out[:, 33:, :2] != 0, axis=2)

    h_reshaped = out[:, 33:].reshape(-1, 2, 21, 3)
    h_vis = np.any(h_reshaped[..., :2] != 0, axis=(2, 3))
    if int((h_vis[:, 0] & h_vis[:, 1]).sum()) >= 4 and rng.random() < 0.18:
        w0_m = (
            np.linalg.norm(np.diff(h_reshaped[h_vis[:, 0], 0, 0, :2], axis=0), axis=1).sum()
            if h_vis[:, 0].sum() > 1
            else 0.0
        )
        w1_m = (
            np.linalg.norm(np.diff(h_reshaped[h_vis[:, 1], 1, 0, :2], axis=0), axis=1).sum()
            if h_vis[:, 1].sum() > 1
            else 0.0
        )
        low_slot = 1 if w1_m <= w0_m else 0
        high_slot = 1 - low_slot
        if min(w0_m, w1_m) < 0.65 * max(w0_m, w1_m, 1e-4):
            drop_mask = h_vis[:, high_slot] & (rng.random(len(out)) < 0.75)
            h_reshaped[drop_mask, low_slot] = 0.0
            out[:, 33:] = h_reshaped.reshape(-1, 42, 3)
            pres_hand = np.any(out[:, 33:, :2] != 0, axis=2)

    valid_sh = pres_pose[:, 11] & pres_pose[:, 12]
    if valid_sh.any():
        center = (out[valid_sh, 11, :2] + out[valid_sh, 12, :2]).mean(axis=0) * 0.5
        sw = float(
            np.linalg.norm(out[valid_sh, 11, :2] - out[valid_sh, 12, :2], axis=1).mean()
        )
        angle = np.deg2rad(rng.uniform(-12.0, 12.0))
        scale = rng.uniform(0.88, 1.12)
        x_stretch = rng.uniform(0.92, 1.08)
        c, s = math.cos(angle), math.sin(angle)

        head_y_scale = float(rng.uniform(0.88, 1.12))
        hand_dx = float(rng.normal(0.0, 0.028 * sw))
        hand_dy = float(rng.normal(0.0, 0.042 * sw))
        palm_scale_factor = float(rng.uniform(0.88, 1.12))

        for sl, pm in [(slice(0, 33), pres_pose), (slice(33, 75), pres_hand)]:
            pts = out[:, sl, :2] - center
            rx = (pts[..., 0] * c - pts[..., 1] * s) * scale * x_stretch
            ry = (pts[..., 0] * s + pts[..., 1] * c) * scale
            jitter = rng.normal(0.0, 0.0025, size=pts.shape)
            out[:, sl, 0] = np.where(pm, rx + center[0] + jitter[..., 0], 0.0)
            out[:, sl, 1] = np.where(pm, ry + center[1] + jitter[..., 1], 0.0)

        nose_vis = pres_pose[:, 0]
        sh_cy = 0.5 * (out[:, 11, 1] + out[:, 12, 1])
        out[nose_vis, 0, 1] = sh_cy[nose_vis] + (
            out[nose_vis, 0, 1] - sh_cy[nose_vis]
        ) * head_y_scale

        for h_base in (33, 54):
            h_sl = slice(h_base, h_base + 21)
            h_pm = pres_hand[:, h_base - 33 : h_base - 33 + 21]
            w_xy = out[:, h_base : h_base + 1, :2]
            rel_xy = (out[:, h_sl, :2] - w_xy) * palm_scale_factor
            out[:, h_sl, 0] = np.where(h_pm, w_xy[..., 0] + rel_xy[..., 0] + hand_dx, 0.0)
            out[:, h_sl, 1] = np.where(h_pm, w_xy[..., 1] + rel_xy[..., 1] + hand_dy, 0.0)
        for el_idx in (13, 14):
            el_vis = pres_pose[:, el_idx]
            out[el_vis, el_idx, 0] += 0.5 * hand_dx
            out[el_vis, el_idx, 1] += 0.5 * hand_dy

    z_scale = (
        float(rng.uniform(-0.35, 1.30))
        if rng.random() < 0.20
        else float(rng.uniform(0.35, 1.30))
    )
    out[:, 33:, 2] = np.where(
        pres_hand,
        out[:, 33:, 2] * z_scale
        + rng.normal(0.0, 0.003, size=out[:, 33:, 2].shape),
        0.0,
    )
    out[..., 0] /= aspect
    return out


def mirror_v3(x: np.ndarray) -> np.ndarray:
    r = x.copy()
    r[..., X_FLIP] *= -1.0
    return r


def _span_q(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.quantile(values, 0.8) - np.quantile(values, 0.2))


def _slot_motion_v3(rows: np.ndarray, slot: int) -> float:
    tips = 0.0
    for j in range(5):
        idx = slot * 42 + (4 + 4 * j) * 2
        x = _span_q(rows[:, idx])
        y = _span_q(rows[:, idx + 1])
        z = _span_q(rows[:, 112 + slot * 5 + j])
        tips += math.sqrt(x * x + y * y + z * z)
    peak = tips / 5.0
    body = rows[rows[:, 111] > 0.95]
    if len(body) >= 8:
        wx = _span_q(body[:, 91 + slot * 10])
        wy = _span_q(body[:, 92 + slot * 10])
        peak = max(peak, math.sqrt(wx * wx + wy * wy))
    return peak


def motion_v3(seq: np.ndarray) -> float:
    """Exact NumPy mirror of SignClassifier._motion for version == 3."""
    best = 0.0
    for two in (False, True):
        rows = seq[seq[:, 88] > 0.95] if two else seq[seq[:, 88] < 0.05]
        if len(rows) < 8:
            continue
        for slot in range(2 if two else 1):
            best = max(best, _slot_motion_v3(rows, slot))
    if best == 0.0 and len(seq) >= 8:
        best = _slot_motion_v3(seq, 0)
    return best


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def forward_v4(x: np.ndarray, params: dict[str, np.ndarray]):
    b_size, steps, feat = x.shape
    h = np.zeros((b_size, HIDDEN), dtype="f")
    projected = (x.reshape(-1, feat) @ params["w"] + params["b"]).reshape(
        b_size, steps, 3 * HIDDEN
    )
    history = []
    for i in range(steps):
        u = h @ params["u"]
        z, r = np.split(
            sigmoid(projected[:, i, : 2 * HIDDEN] + u[:, : 2 * HIDDEN]), 2, axis=1
        )
        n = np.tanh(projected[:, i, 2 * HIDDEN :] + r * u[:, 2 * HIDDEN :])
        history.append((h, z, r, n, u[:, 2 * HIDDEN :]))
        h = z * h + (1.0 - z) * n
    summary = np.concatenate([h, x.mean(axis=1)], axis=1)
    return summary @ params["out"] + params["bias"], (summary, history)


def predict_probs(x: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    logits = 0.5 * (forward_v4(x, params)[0] + forward_v4(mirror_v3(x), params)[0])
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    return logits, probs


def evaluate_split(
    x: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray], motions: np.ndarray | None = None
) -> dict:
    logits, probs = predict_probs(x, params)
    preds = probs.argmax(axis=1)
    scores = probs[np.arange(len(y)), preds]
    top3 = np.any(np.argsort(-probs, axis=1)[:, :3] == y[:, None], axis=1)
    top5 = np.any(np.argsort(-probs, axis=1)[:, :5] == y[:, None], axis=1)
    nll = float(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-9)).mean())
    correct = preds == y
    return {
        "count": int(len(y)),
        "top1": float(correct.mean()),
        "top3": float(top3.mean()),
        "top5": float(top5.mean()),
        "nll": nll,
        "mean_score_correct": float(scores[correct].mean()) if correct.any() else 0.0,
        "mean_score_wrong": float(scores[~correct].mean()) if (~correct).any() else 0.0,
        "preds": preds,
        "scores": scores,
        "probs": probs,
        "logits": logits,
    }


def train_gru(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    num_classes: int,
    epochs: int = 65,
    wd: float = 2e-4,
    feat_drop: float = 0.12,
    smooth: float = 0.04,
):
    rng = np.random.default_rng(37)

    def init_w(a: int, b: int) -> np.ndarray:
        return rng.normal(0.0, math.sqrt(2.0 / (a + b)), (a, b)).astype("f")

    p = {
        "w": init_w(FEATURES, 3 * HIDDEN),
        "u": init_w(HIDDEN, 3 * HIDDEN),
        "b": np.zeros(3 * HIDDEN, "f"),
        "out": init_w(HIDDEN + FEATURES, num_classes),
        "bias": np.zeros(num_classes, "f"),
    }
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(t) for k, t in p.items()}
    step = 0
    best_key = (-1.0, -999.0)
    best_epoch = 0
    best_p = None

    for epoch in range(epochs):
        indices = rng.permutation(len(x_tr))
        for start in range(0, len(x_tr), 64):
            batch = indices[start : start + 64]
            a = x_tr[batch].copy()
            a[::2] = mirror_v3(a[::2])
            if feat_drop > 0:
                mask = (
                    rng.random((a.shape[0], 1, a.shape[2])) >= feat_drop
                ).astype("f") / (1.0 - feat_drop)
                a *= mask
            logits, (summary, history) = forward_v4(a, p)
            prob = np.exp(logits - logits.max(axis=1, keepdims=True))
            prob = prob / prob.sum(axis=1, keepdims=True)
            d = prob - (smooth / num_classes)
            d[np.arange(len(d)), y_tr[batch]] -= 1.0 - smooth
            d /= len(d)
            g = {k: np.zeros_like(t) for k, t in p.items()}
            g["out"] = summary.T @ d
            g["bias"] = d.sum(axis=0)
            dh = (d @ p["out"].T)[:, :HIDDEN]
            dx_all = np.empty((len(a), STEPS, 3 * HIDDEN), dtype="f")
            for i in range(STEPS - 1, -1, -1):
                prev, z, r, n, un = history[i]
                dn = dh * (1.0 - z) * (1.0 - n * n)
                dz = dh * (prev - n) * z * (1.0 - z)
                dr = dn * un * r * (1.0 - r)
                dx_all[:, i, :HIDDEN] = dz
                dx_all[:, i, HIDDEN : 2 * HIDDEN] = dr
                dx_all[:, i, 2 * HIDDEN :] = dn
                du = np.concatenate([dz, dr, dn * r], axis=1)
                g["u"] += prev.T @ du
                dh = dh * z + du @ p["u"].T
            g["w"] = a.reshape(-1, FEATURES).T @ dx_all.reshape(-1, 3 * HIDDEN)
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
            ev_val = evaluate_split(x_val, y_val, p)
            ev_te = evaluate_split(x_te, y_te, p)
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
    print(f"Selected best validation checkpoint from epoch {best_epoch}")
    return best_p, best_epoch


def compute_motion_floors(
    train_seqs_12fps: list[np.ndarray],
    train_labels: list[int],
    idx_to_key: list[str],
) -> dict[str, float]:
    """Conservative motion floors across all active classes to block frozen frames."""
    by_cls: dict[int, list[float]] = {i: [] for i in range(len(idx_to_key))}
    for seq, cls in zip(train_seqs_12fps, train_labels):
        by_cls[cls].append(motion_v3(seq))

    floors: dict[str, float] = {}
    for cls, vals in by_cls.items():
        if len(vals) < 10:
            continue
        arr = np.asarray(vals, dtype="f")
        min_v = float(arr.min())
        p5_v = float(np.percentile(arr, 5))
        if min_v > 0.05:
            floor = min(0.30, max(0.045, round(min(p5_v * 0.55, min_v * 0.65), 5)))
            floors[idx_to_key[cls]] = floor
    return floors


def calibrate_threshold(ev_val: dict) -> tuple[float, list[dict]]:
    """Select threshold using strictly validation split to avoid Signer 4 leakage."""
    preds = ev_val["preds"]
    scores = ev_val["scores"]
    # Note: val comes from Signers 1-3 where probabilities run higher than unseen Signer 4.
    # Target high coverage while filtering genuinely uncertain predictions.
    sweep = []
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        mask = scores >= thr
        cov = float(mask.mean())
        acc_on_acc = (
            float((preds[mask] == ev_val["y"][mask]).mean()) if mask.any() else 0.0
        )
        sweep.append(
            {
                "threshold": thr,
                "coverage": cov,
                "accepted_accuracy": acc_on_acc,
            }
        )
    # Pick 0.55 if coverage >= 0.94 and accepted accuracy improves, else highest threshold with >= 0.96 coverage.
    chosen = 0.55
    for entry in sweep:
        if entry["threshold"] <= 0.60 and entry["coverage"] >= 0.95:
            chosen = entry["threshold"]
    return float(chosen), sweep


def main() -> None:
    t_start = time.time()
    labels_info = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    key_to_idx = {r["key"]: int(r["model_index"]) for r in labels_info}
    idx_to_key = [r["key"] for r in labels_info]
    num_classes = len(labels_info)
    assert num_classes == 104, f"Expected 104 active classes, got {num_classes}"
    assert 2 not in {r["id"] for r in labels_info}, "ID 2 must be excluded"

    rng = np.random.default_rng(2026)
    x_tr, y_tr = [], []
    x_val, y_val = [], []
    x_te, y_te = [], []
    te_clip_meta = []
    tv_12fps_seqs: list[np.ndarray] = []
    tv_12fps_labels: list[int] = []

    with np.load(CLIPS_PATH) as d:
        raw_all = d["raw_landmarks"]
        offsets = d["clip_offsets"]
        m_idx = d["model_indices"]
        c_ids = d["clip_ids"]
        aspects = d["aspects"]
        fps_arr = d["fps"]
        splits = d["splits"]

    # Remember clip raw slices for clean held-out sample export later.
    test_clip_indices: list[int] = []

    for k in range(len(m_idx)):
        raw = raw_all[offsets[k] : offsets[k + 1]]
        idx = int(m_idx[k])
        aspect = float(aspects[k])
        fps = float(fps_arr[k])
        split = str(splits[k])

        if split == "train":
            stride_12 = max(1, round(fps / 12.0))
            base_12 = encode_v4(raw[::stride_12], aspect)
            if base_12 is not None:
                tv_12fps_seqs.append(base_12)
                tv_12fps_labels.append(idx)

            for rate in (18.0, 12.0, 9.0, 6.0, 4.5):
                stride = max(1, round(fps / rate))
                sub = raw[int(rng.integers(stride)) :: stride]
                if rate in (12.0, 6.0):
                    sub = augment_raw(sub, aspect, rng, trim=(rate == 6.0))
                if rate == 9.0 and len(sub) > 6:
                    keep = rng.random(len(sub)) > 0.15
                    if int(keep.sum()) >= 3:
                        sub = sub[keep]
                feat = encode_v4(sub, aspect)
                if feat is not None:
                    x_tr.append(feat)
                    y_tr.append(idx)

            for extra_rate, do_trim in ((12.0, True), (9.0, False), (15.0, True)):
                stride = max(1, round(fps / extra_rate))
                sub = augment_raw(
                    raw[int(rng.integers(stride)) :: stride],
                    aspect,
                    rng,
                    trim=do_trim,
                )
                feat = encode_v4(sub, aspect)
                if feat is not None:
                    x_tr.append(feat)
                    y_tr.append(idx)
        else:
            stride = max(1, round(fps / 12.0))
            feat = encode_v4(raw[::stride], aspect)
            if feat is None:
                continue
            if split == "val":
                x_val.append(feat)
                y_val.append(idx)
                tv_12fps_seqs.append(feat)
                tv_12fps_labels.append(idx)
            else:
                x_te.append(feat)
                y_te.append(idx)
                te_clip_meta.append(
                    {
                        "npz_idx": k,
                        "clip_id": int(c_ids[k]),
                        "model_idx": idx,
                        "aspect": aspect,
                        "fps": fps,
                        "stride": stride,
                    }
                )
                test_clip_indices.append(k)

    # Add earlier greeting_keypoints strictly on train clips (clip < 15 and clip % 5 != 4).
    greeting_added = 0
    if GREETING_KEYPOINTS_DIR.is_dir():
        for gdir in sorted(GREETING_KEYPOINTS_DIR.iterdir()):
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
                    feat = encode_v4(
                        g_raw[int(rng.integers(stride)) :: stride], g_aspect
                    )
                    if feat is not None:
                        x_tr.append(feat)
                        y_tr.append(idx)
                        greeting_added += 1

    x_tr_arr = np.asarray(x_tr, dtype="f")
    y_tr_arr = np.asarray(y_tr, dtype=np.int64)
    x_val_arr = np.asarray(x_val, dtype="f")
    y_val_arr = np.asarray(y_val, dtype=np.int64)
    x_te_arr = np.asarray(x_te, dtype="f")
    y_te_arr = np.asarray(y_te, dtype=np.int64)

    print(
        f"Encoded in {time.time() - t_start:.1f}s: "
        f"train={x_tr_arr.shape} (greeting_extra={greeting_added}), "
        f"val={x_val_arr.shape}, test={x_te_arr.shape}"
    )

    recalibrate = "--recalibrate" in sys.argv
    if recalibrate and MODEL_OUT_PATH.is_file():
        existing = json.loads(MODEL_OUT_PATH.read_text(encoding="utf-8"))
        best_p = {
            "w": np.asarray(existing["w"], dtype="f").reshape(FEATURES, 3 * HIDDEN),
            "u": np.asarray(existing["u"], dtype="f").reshape(HIDDEN, 3 * HIDDEN),
            "b": np.asarray(existing["b"], dtype="f"),
            "out": np.asarray(existing["out"], dtype="f").reshape(
                HIDDEN + FEATURES, num_classes
            ),
            "bias": np.asarray(existing["bias"], dtype="f"),
        }
        best_epoch = int(
            json.loads(METRICS_OUT_PATH.read_text(encoding="utf-8"))["model"][
                "best_val_epoch"
            ]
        )
        print(f"Loaded existing trained weights (epoch {best_epoch}) for recalibration")
    else:
        best_p, best_epoch = train_gru(
            x_tr_arr,
            y_tr_arr,
            x_val_arr,
            y_val_arr,
            x_te_arr,
            y_te_arr,
            num_classes,
            epochs=65,
            wd=2.5e-4,
            feat_drop=0.14,
            smooth=0.04,
        )

    ev_val = evaluate_split(x_val_arr, y_val_arr, best_p)
    ev_val["y"] = y_val_arr
    ev_te = evaluate_split(x_te_arr, y_te_arr, best_p)
    ev_te["y"] = y_te_arr

    motion_floors = compute_motion_floors(tv_12fps_seqs, tv_12fps_labels, idx_to_key)
    threshold, val_sweep = calibrate_threshold(ev_val)

    # Mirror invariance check on held-out Signer 4 test split.
    _, probs_orig = predict_probs(x_te_arr, best_p)
    _, probs_mir = predict_probs(mirror_v3(x_te_arr), best_p)
    mirror_max_diff = float(np.abs(probs_orig - probs_mir).max())

    # Apply Dart-equivalent gating (threshold + motionFloors) to test predictions.
    te_motions = np.asarray([motion_v3(seq) for seq in x_te_arr], dtype="f")
    accepted_mask = np.zeros(len(y_te_arr), dtype=bool)
    accepted_correct = np.zeros(len(y_te_arr), dtype=bool)
    motion_rejected = 0
    for i in range(len(y_te_arr)):
        pred_lbl = idx_to_key[int(ev_te["preds"][i])]
        score = float(ev_te["scores"][i])
        floor = motion_floors.get(pred_lbl)
        if floor is not None and float(te_motions[i]) < floor:
            motion_rejected += 1
            continue
        if score >= threshold:
            accepted_mask[i] = True
            accepted_correct[i] = int(ev_te["preds"][i]) == int(y_te_arr[i])

    # Simulated static frozen-frame false-trigger check: repeat mid-clip frame 16 times.
    frozen_x = np.repeat(x_te_arr[:, STEPS // 2 : STEPS // 2 + 1, :STATIC_FEATURES], STEPS, axis=1)
    frozen_full = np.concatenate(
        [frozen_x, np.zeros((len(frozen_x), STEPS, len(VEL_INDICES)), dtype="f")], axis=2
    )
    ev_frozen = evaluate_split(frozen_full, y_te_arr, best_p)
    frozen_accepted = 0
    for i in range(len(y_te_arr)):
        pred_lbl = idx_to_key[int(ev_frozen["preds"][i])]
        floor = motion_floors.get(pred_lbl)
        if floor is not None and motion_v3(frozen_full[i]) < floor:
            continue
        if float(ev_frozen["scores"][i]) >= threshold:
            frozen_accepted += 1

    # Confusion matrix & per-class metrics on Signer 4 test split.
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t_cls, p_cls in zip(y_te_arr, ev_te["preds"]):
        cm[int(t_cls), int(p_cls)] += 1

    per_class = []
    f1s = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        cls_mask = y_te_arr == c
        per_class.append(
            {
                "id": labels_info[c]["id"],
                "model_index": c,
                "key": idx_to_key[c],
                "category": labels_info[c]["category"],
                "test_count": int(cls_mask.sum()),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "accepted_recall": round(
                    float(accepted_correct[cls_mask].sum() / max(1, cls_mask.sum())), 4
                ),
            }
        )

    confused_pairs = []
    for r in range(num_classes):
        for c in range(num_classes):
            if r != c and cm[r, c] > 0:
                confused_pairs.append(
                    {
                        "true_label": idx_to_key[r],
                        "pred_label": idx_to_key[c],
                        "count": int(cm[r, c]),
                    }
                )
    confused_pairs.sort(key=lambda item: (-item["count"], item["true_label"]))

    test_sweep = []
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = ev_te["scores"] >= thr
        test_sweep.append(
            {
                "threshold": thr,
                "coverage": round(float(mask.mean()), 4),
                "accepted_accuracy": round(
                    float((ev_te["preds"][mask] == y_te_arr[mask]).mean())
                    if mask.any()
                    else 0.0,
                    4,
                ),
            }
        )

    # Write confusion_matrix.csv
    with CONFUSION_OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred"] + idx_to_key)
        for r in range(num_classes):
            writer.writerow([idx_to_key[r]] + cm[r].tolist())

    # Export model JSON matching SignClassifier.decode schema
    model_payload = {
        "version": VERSION,
        "steps": STEPS,
        "features": FEATURES,
        "hidden": HIDDEN,
        "labels": idx_to_key,
        "labelIds": [int(r["id"]) for r in labels_info],
        "threshold": round(threshold, 4),
        "motionFloors": motion_floors,
        "w": np.round(best_p["w"].ravel(), 6).tolist(),
        "u": np.round(best_p["u"].ravel(), 6).tolist(),
        "b": np.round(best_p["b"].ravel(), 6).tolist(),
        "out": np.round(best_p["out"].ravel(), 6).tolist(),
        "bias": np.round(best_p["bias"].ravel(), 6).tolist(),
    }
    MODEL_OUT_PATH.write_text(
        json.dumps(model_payload, separators=(",", ":")), encoding="utf-8"
    )

    # Export held-out test_samples.json (one Signer 4 clip per class) using exact Dart input geometry.
    with np.load(CLIPS_PATH) as d:
        raw_all = d["raw_landmarks"]
        offsets = d["clip_offsets"]

    chosen_by_cls: dict[int, int] = {}
    for pos, meta in enumerate(te_clip_meta):
        cls = meta["model_idx"]
        # Prefer clips predicted correctly and valid after encoding.
        if cls not in chosen_by_cls or (
            int(ev_te["preds"][pos]) == cls
            and int(ev_te["preds"][chosen_by_cls[cls]]) != cls
        ):
            chosen_by_cls[cls] = pos

    samples_out = []
    for cls in sorted(chosen_by_cls):
        pos = chosen_by_cls[cls]
        meta = te_clip_meta[pos]
        raw = raw_all[offsets[meta["npz_idx"]] : offsets[meta["npz_idx"] + 1]]
        strided = raw[:: meta["stride"]]
        frames_out = []
        for frame in strided:
            pose5 = [
                [round(float(frame[p_idx, 0]), 6), round(float(frame[p_idx, 1]), 6)]
                for p_idx in POSE_ANCHORS
            ]
            h0 = [
                [
                    round(float(frame[33 + j, 0]), 6),
                    round(float(frame[33 + j, 1]), 6),
                    round(float(frame[33 + j, 2]), 6),
                ]
                for j in range(21)
            ]
            h1 = [
                [
                    round(float(frame[54 + j, 0]), 6),
                    round(float(frame[54 + j, 1]), 6),
                    round(float(frame[54 + j, 2]), 6),
                ]
                for j in range(21)
            ]
            frames_out.append({"pose": pose5, "hand0": h0, "hand1": h1})
        top3_idx = np.argsort(-ev_te["probs"][pos])[:3]
        samples_out.append(
            {
                "id": int(labels_info[cls]["id"]),
                "label": idx_to_key[cls],
                "clip_id": meta["clip_id"],
                "aspect": round(float(meta["aspect"]), 6),
                "fps": round(float(meta["fps"]), 3),
                "expected_pred": idx_to_key[int(ev_te["preds"][pos])],
                "expected_score": round(float(ev_te["scores"][pos]), 6),
                "expected_motion": round(float(te_motions[pos]), 6),
                "top3": [
                    [idx_to_key[int(ti)], round(float(ev_te["probs"][pos, ti]), 6)]
                    for ti in top3_idx
                ],
                "frames": frames_out,
            }
        )
    SAMPLES_OUT_PATH.write_text(
        json.dumps(samples_out, separators=(",", ":")), encoding="utf-8"
    )

    greetings_keys = [
        "GOOD_MORNING",
        "GOOD_AFTERNOON",
        "HELLO",
        "HOW_ARE_YOU",
        "IM_FINE",
        "NICE_TO_MEET_YOU",
        "THANK_YOU",
        "YOURE_WELCOME",
        "SEE_YOU_TOMORROW",
    ]
    greeting_metrics = [
        row for row in per_class if row["key"] in set(greetings_keys)
    ]

    metrics_payload = {
        "model": {
            "version": VERSION,
            "steps": STEPS,
            "features": FEATURES,
            "hidden": HIDDEN,
            "num_classes": num_classes,
            "excluded_ids": [2],
            "best_val_epoch": best_epoch,
            "threshold": round(threshold, 4),
            "motion_floors_count": len(motion_floors),
            "model_size_bytes": MODEL_OUT_PATH.stat().st_size,
        },
        "validation_split": {
            "count": ev_val["count"],
            "top1": round(ev_val["top1"], 4),
            "top3": round(ev_val["top3"], 4),
            "top5": round(ev_val["top5"], 4),
            "nll": round(ev_val["nll"], 4),
            "threshold_sweep": val_sweep,
        },
        "test_signer4": {
            "count": ev_te["count"],
            "top1": round(ev_te["top1"], 4),
            "top3": round(ev_te["top3"], 4),
            "top5": round(ev_te["top5"], 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "nll": round(ev_te["nll"], 4),
            "mean_score_correct": round(ev_te["mean_score_correct"], 4),
            "mean_score_wrong": round(ev_te["mean_score_wrong"], 4),
            "accepted_coverage": round(float(accepted_mask.mean()), 4),
            "accepted_accuracy": round(
                float(accepted_correct[accepted_mask].mean())
                if accepted_mask.any()
                else 0.0,
                4,
            ),
            "motion_rejected_count": int(motion_rejected),
            "mirror_max_prob_diff": round(mirror_max_diff, 8),
            "frozen_static_false_accepts": int(frozen_accepted),
            "threshold_sweep": test_sweep,
        },
        "greetings_test": greeting_metrics,
        "top_confused_pairs": confused_pairs[:25],
        "lowest_f1_classes": sorted(per_class, key=lambda r: (r["f1"], r["key"]))[:15],
        "per_class_test": per_class,
    }
    METRICS_OUT_PATH.write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    print(
        f"\nSaved {MODEL_OUT_PATH.name} ({MODEL_OUT_PATH.stat().st_size / 1024:.1f} KB), "
        f"threshold={threshold}, motionFloors={len(motion_floors)}"
    )
    print(
        f"Validation: top1={ev_val['top1']:.4f}, top3={ev_val['top3']:.4f} | "
        f"Signer4 Test: top1={ev_te['top1']:.4f}, top3={ev_te['top3']:.4f}, "
        f"macroF1={np.mean(f1s):.4f}"
    )
    print(
        f"At threshold={threshold}: test coverage={accepted_mask.mean():.4f}, "
        f"accepted_acc={(accepted_correct[accepted_mask].mean() if accepted_mask.any() else 0):.4f}, "
        f"motion_rejected={motion_rejected}, frozen_false_accepts={frozen_accepted}/{len(y_te_arr)}"
    )
    print("Greetings:", [(g["key"], g["recall"], g["precision"]) for g in greeting_metrics])
    print("Top 10 confused:", confused_pairs[:10])


if __name__ == "__main__":
    main()
