"""Extract MediaPipe hand keypoints and 88-dim features from the Collated FSL alphabet dataset."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLATED = ROOT / "Collated"
DEFAULT_OUT = ROOT / "alphabet_keypoints"
DEFAULT_HAND_TASK = Path(
    r"C:\Users\Hans\AppData\Local\Pub\Cache\hosted\pub.dev\hand_landmarker-2.2.0\android\src\main\assets\hand_landmarker.task"
)

THUMB_TO_FINGER_TARGETS = (5, 6, 8, 9, 10, 12, 13, 14, 17, 20)
FINGERTIPS = (4, 8, 12, 16, 20)
ADJACENT_TIP_PAIRS = ((8, 12), (12, 16), (16, 20), (4, 16))
CURL_CHAINS = (
    (1, 2, 2, 4),
    (5, 6, 6, 8),
    (9, 10, 10, 12),
    (13, 14, 14, 16),
    (17, 18, 18, 20),
)
FEATURE_DIM = 88

_WORKER_LANDMARKER = None


def _init_worker(hand_task_path: str) -> None:
    global _WORKER_LANDMARKER
    cv2.setNumThreads(1)
    vision = mp.tasks.vision
    base = mp.tasks.BaseOptions
    opts = vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=hand_task_path),
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    _WORKER_LANDMARKER = vision.HandLandmarker.create_from_options(opts)


def compute_features_from_upright(pts: np.ndarray) -> np.ndarray | None:
    """Compute the 88-dim feature vector from (21, 3) upright image-height coordinates."""
    if pts.shape != (21, 3) or not np.isfinite(pts).all():
        return None
    wrist = pts[0]
    local = pts - wrist[None, :]
    palm = float(np.mean(np.linalg.norm(local[[5, 9, 13, 17], :2], axis=-1)))
    if not np.isfinite(palm) or palm <= 1e-4:
        return None

    norm = np.clip(local / palm, -5.0, 5.0).astype(np.float32)
    xy = norm[:, :2]
    feats = np.empty(FEATURE_DIM, dtype=np.float32)
    feats[:63] = norm.reshape(-1)

    idx = 63
    for j in THUMB_TO_FINGER_TARGETS:
        feats[idx] = float(np.linalg.norm(xy[4] - xy[j]))
        idx += 1
    for j in FINGERTIPS:
        feats[idx] = float(np.linalg.norm(xy[j]))
        idx += 1
    for a, b in ADJACENT_TIP_PAIRS:
        feats[idx] = float(np.linalg.norm(xy[a] - xy[b]))
        idx += 1
    for a, b, c, d in CURL_CHAINS:
        u = xy[b] - xy[a]
        v = xy[d] - xy[c]
        denom = max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-6)
        feats[idx] = float(np.dot(u, v) / denom)
        idx += 1

    u2 = xy[8] - xy[12]
    v2 = xy[5] - xy[9]
    denom2 = max(float(np.linalg.norm(u2) * np.linalg.norm(v2)), 1e-6)
    feats[idx] = float(np.dot(u2, v2) / denom2)
    return feats


def _select_primary_hand(
    hand_landmarks_list,
    width: int,
    height: int,
    pad_x: int = 0,
    pad_y: int = 0,
) -> tuple[np.ndarray, int] | None:
    """Return (21, 3) upright image-height coords for the topmost valid hand."""
    padded_w = width + 2 * pad_x
    padded_h = height + 2 * pad_y
    aspect = width / height
    candidates = []
    for hand_lms in hand_landmarks_list:
        if len(hand_lms) != 21:
            continue
        pts = np.empty((21, 3), dtype=np.float32)
        for j, lm in enumerate(hand_lms):
            x_orig = (lm.x * padded_w - pad_x) / width
            y_orig = (lm.y * padded_h - pad_y) / height
            z_orig = lm.z * (padded_w / width)
            pts[j, 0] = x_orig * aspect
            pts[j, 1] = y_orig
            pts[j, 2] = z_orig * aspect
        wrist = pts[0]
        palm = float(np.mean(np.linalg.norm(pts[[5, 9, 13, 17], :2] - wrist[:2], axis=-1)))
        if np.isfinite(pts).all() and palm > 1e-4:
            candidates.append((float(wrist[1]), pts))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], len(candidates)


def _process_one(task: tuple[int, str, str]) -> dict:
    sample_idx, label, file_path = task
    path = Path(file_path)
    raw_bytes = path.read_bytes()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {
            "index": sample_idx,
            "ok": False,
            "reason": "decode_error",
            "label": label,
            "path": file_path,
        }
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _WORKER_LANDMARKER.detect(mp_img)
    selected = _select_primary_hand(res.hand_landmarks, w, h)
    used_padding = False

    if selected is None:
        pad_y = max(16, round(h * 0.25))
        pad_x = max(16, round(w * 0.25))
        padded_rgb = cv2.copyMakeBorder(
            rgb, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
        )
        mp_padded = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded_rgb)
        res_pad = _WORKER_LANDMARKER.detect(mp_padded)
        selected = _select_primary_hand(res_pad.hand_landmarks, w, h, pad_x=pad_x, pad_y=pad_y)
        used_padding = True

    if selected is None:
        return {
            "index": sample_idx,
            "ok": False,
            "reason": "no_hand_detected",
            "label": label,
            "path": file_path,
            "width": w,
            "height": h,
        }

    pts, num_hands = selected
    feats = compute_features_from_upright(pts)
    if feats is None:
        return {
            "index": sample_idx,
            "ok": False,
            "reason": "invalid_landmarks",
            "label": label,
            "path": file_path,
            "width": w,
            "height": h,
        }

    return {
        "index": sample_idx,
        "ok": True,
        "pts": pts,
        "feats": feats,
        "num_hands": num_hands,
        "used_padding": used_padding,
        "width": w,
        "height": h,
    }


def _parse_provenance(label: str, path: Path) -> tuple[str, str, tuple]:
    """Return (source_family, initial_group_id, sort_key) for leak-free splitting."""
    stem = path.stem
    if ".rf." in stem:
        base = stem.split(".rf.")[0]
        m = re.match(r"^(\d+)(?:-\d+-|-Copy)?_jpg$", base, re.I)
        if m:
            num = int(m.group(1))
            return "roboflow_or_plain", f"rf_{num}", (0, num, path.name)
        return "roboflow_or_plain", f"rf_{base}", (1, base, path.name)

    if stem.startswith("Image_"):
        m = re.match(r"^Image_([0-9.]+)", stem)
        if m:
            ts = float(m.group(1))
            return "timestamp_burst", f"ts_{ts:.6f}", (ts, path.name)

    m_letter = re.match(
        rf"^({re.escape(label)})(_?)(\d+)(?:-\d+-|\s*\(\d+\)|\s*-\s*Copy)*$",
        stem,
        re.I,
    )
    if m_letter:
        sep = m_letter.group(2) or "none"
        num = int(m_letter.group(3))
        block = num // 10
        return f"letter_{sep}", f"letter_{sep}_{block:04d}", (num, path.name)

    m_plain = re.match(
        r"^(\d+)(?:-\d+-|\s*\(\d+\)|\s*-?\s*Copy)*$",
        stem,
        re.I,
    )
    if m_plain:
        num = int(m_plain.group(1))
        return "roboflow_or_plain", f"rf_{num}", (0, num, path.name)

    return "other", f"other_{stem}", (0, stem, path.name)


def _assign_timestamp_clusters(records: list[dict]) -> None:
    """Cluster timestamp_burst records per class by chronological gaps."""
    by_class: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in records:
        if rec["source_family"] == "timestamp_burst":
            by_class[rec["label"]].append(rec)

    for label, items in by_class.items():
        items.sort(key=lambda r: r["sort_key"][0])
        if len(items) <= 6:
            for idx, r in enumerate(items):
                r["group_id"] = f"{label}:ts_g{idx:03d}"
            continue

        times = [r["sort_key"][0] for r in items]
        gaps = [(times[i + 1] - times[i], i + 1) for i in range(len(times) - 1)]
        split_points = sorted(idx for gap, idx in gaps if gap > 0.5)
        if len(split_points) < 5:
            # Ensure enough contiguous temporal clusters for 70/15/15 splitting
            ranked = sorted(gaps, key=lambda pair: pair[0], reverse=True)
            chosen = set(split_points)
            for _, idx in ranked:
                if len(chosen) >= 6:
                    break
                chosen.add(idx)
            split_points = sorted(chosen)

        bounds = [0] + split_points + [len(items)]
        for cluster_idx in range(len(bounds) - 1):
            start, end = bounds[cluster_idx], bounds[cluster_idx + 1]
            for pos in range(start, end):
                items[pos]["group_id"] = f"{label}:ts_c{cluster_idx:03d}"


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _merge_near_duplicate_groups(records: list[dict], feats: np.ndarray) -> int:
    """Merge within-class groups whose 63D landmark distance is < 0.04."""
    by_class: dict[str, list[int]] = collections.defaultdict(list)
    for i, rec in enumerate(records):
        by_class[rec["label"]].append(i)

    merges = 0
    for label, indices in by_class.items():
        if len(indices) < 2:
            continue
        uf = _UnionFind(len(indices))
        group_to_rep: dict[str, int] = {}
        for local_i, global_i in enumerate(indices):
            gid = records[global_i]["group_id"]
            if gid in group_to_rep:
                uf.union(group_to_rep[gid], local_i)
            else:
                group_to_rep[gid] = local_i

        coords = feats[indices, :63]
        # Compute pairwise distances in chunks to keep memory small
        n = len(indices)
        for start in range(0, n, 128):
            end = min(n, start + 128)
            block = coords[start:end]
            diff = block[:, None, :] - coords[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            rows, cols = np.where(dists < 0.04)
            for r, c in zip(rows.tolist(), cols.tolist()):
                g_a = start + r
                g_b = c
                if g_a < g_b and uf.find(g_a) != uf.find(g_b):
                    uf.union(g_a, g_b)
                    merges += 1

        for local_i, global_i in enumerate(indices):
            root = uf.find(local_i)
            records[global_i]["merged_group_id"] = f"{label}:m{root:04d}"
    return merges


def _assign_splits(records: list[dict]) -> dict[str, int]:
    """Assign contiguous chronological/numeric group splits (70% train, 15% val, 15% test)."""
    by_class_family: dict[tuple[str, str], dict[str, list[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for rec in records:
        fam = rec["source_family"]
        gid = rec["merged_group_id"]
        by_class_family[(rec["label"], fam)][gid].append(rec)

    split_counts = collections.Counter()
    for (label, fam), groups in sorted(by_class_family.items()):
        ordered_groups = sorted(
            groups.items(),
            key=lambda kv: min(r["sort_key"] for r in kv[1]),
        )
        total_samples = sum(len(g[1]) for g in ordered_groups)
        train_target = round(total_samples * 0.70)
        val_target = round(total_samples * 0.85)
        cum = 0
        for g_idx, (gid, grp_recs) in enumerate(ordered_groups):
            if g_idx == 0:
                split = "train"
            elif g_idx == len(ordered_groups) - 1 and len(ordered_groups) >= 3:
                split = "test"
            elif cum < train_target:
                split = "train"
            elif cum < val_target:
                split = "val"
            else:
                split = "test"
            for r in grp_recs:
                r["split"] = split
                split_counts[split] += 1
            cum += len(grp_recs)

    # Second pass: ensure every merged_group_id has a single split (in case cross-family merge occurred)
    group_split: dict[str, str] = {}
    priority = {"test": 2, "val": 1, "train": 0}
    for rec in records:
        gid = rec["merged_group_id"]
        cur = group_split.get(gid)
        if cur is None or priority[rec["split"]] > priority[cur]:
            group_split[gid] = rec["split"]

    final_counts = collections.Counter()
    for rec in records:
        rec["split"] = group_split[rec["merged_group_id"]]
        final_counts[rec["split"]] += 1
    return dict(final_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collated", type=Path, default=DEFAULT_COLLATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hand-task", type=Path, default=DEFAULT_HAND_TASK)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not args.collated.is_dir():
        raise FileNotFoundError(f"Collated directory not found: {args.collated}")
    if not args.hand_task.is_file():
        raise FileNotFoundError(f"hand_landmarker.task not found: {args.hand_task}")

    args.output.mkdir(parents=True, exist_ok=True)
    classes = sorted(d.name for d in args.collated.iterdir() if d.is_dir())

    # 1. Recursive inventory and exact SHA-256 deduplication
    seen_hashes: dict[str, tuple[str, str]] = {}
    unique_records: list[dict] = []
    exact_duplicates = 0
    corrupted_files: list[dict] = []
    image_sizes = collections.Counter()
    per_class_raw = collections.Counter()
    per_class_exact_dups = collections.Counter()

    for label in classes:
        files = sorted(f for f in (args.collated / label).rglob("*") if f.is_file())
        for f in files:
            per_class_raw[label] += 1
            data = f.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if sha in seen_hashes:
                exact_duplicates += 1
                per_class_exact_dups[label] += 1
                continue
            seen_hashes[sha] = (label, str(f))
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    image_sizes[f"{im.size[0]}x{im.size[1]}"] += 1
            except Exception as exc:
                corrupted_files.append({"label": label, "path": str(f), "error": str(exc)})
                continue

            fam, gid, sort_key = _parse_provenance(label, f)
            unique_records.append(
                {
                    "label": label,
                    "path": str(f),
                    "rel_path": f.relative_to(args.collated).as_posix(),
                    "sha256": sha,
                    "source_family": fam,
                    "group_id": f"{label}:{gid}",
                    "sort_key": sort_key,
                }
            )

    _assign_timestamp_clusters(unique_records)

    print(
        f"Inventory: {sum(per_class_raw.values())} total files across {len(classes)} classes; "
        f"exact SHA256 duplicates skipped={exact_duplicates}, corrupted={len(corrupted_files)}, "
        f"unique candidates={len(unique_records)}",
        flush=True,
    )

    # 2. Extract MediaPipe landmarks in parallel
    tasks = [(i, rec["label"], rec["path"]) for i, rec in enumerate(unique_records)]
    results = [None] * len(tasks)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.hand_task),),
    ) as pool:
        for done_count, res in enumerate(pool.map(_process_one, tasks, chunksize=32), 1):
            results[res["index"]] = res
            if done_count % 1000 == 0 or done_count == len(tasks):
                print(f"  processed {done_count}/{len(tasks)}", flush=True)

    valid_records = []
    valid_pts = []
    valid_feats = []
    num_hands_list = []
    padding_used_count = 0
    rejected_by_class = collections.Counter()
    rejected_reasons = collections.Counter()
    rejected_samples = []
    two_hand_by_class = collections.Counter()

    for rec, res in zip(unique_records, results):
        if not res["ok"]:
            rejected_by_class[rec["label"]] += 1
            rejected_reasons[res["reason"]] += 1
            if len(rejected_samples) < 50:
                rejected_samples.append({"label": rec["label"], "rel_path": rec["rel_path"], "reason": res["reason"]})
            continue
        if res["used_padding"]:
            padding_used_count += 1
        if res["num_hands"] > 1:
            two_hand_by_class[rec["label"]] += 1
        valid_records.append(rec)
        valid_pts.append(res["pts"])
        valid_feats.append(res["feats"])
        num_hands_list.append(res["num_hands"])

    pts_arr = np.asarray(valid_pts, dtype=np.float32)
    feats_arr = np.asarray(valid_feats, dtype=np.float32)
    num_hands_arr = np.asarray(num_hands_list, dtype=np.int32)

    merges = _merge_near_duplicate_groups(valid_records, feats_arr)
    split_counts = _assign_splits(valid_records)

    labels_arr = np.asarray([classes.index(r["label"]) for r in valid_records], dtype=np.int32)
    splits_arr = np.asarray([r["split"] for r in valid_records])
    groups_arr = np.asarray([r["merged_group_id"] for r in valid_records])
    paths_arr = np.asarray([r["rel_path"] for r in valid_records])
    families_arr = np.asarray([r["source_family"] for r in valid_records])

    npz_path = args.output / "alphabet_features.npz"
    np.savez_compressed(
        npz_path,
        X=feats_arr,
        landmarks=pts_arr,
        y=labels_arr,
        classes=np.asarray(classes),
        split=splits_arr,
        group_id=groups_arr,
        rel_path=paths_arr,
        source_family=families_arr,
        num_hands=num_hands_arr,
    )

    per_class_valid = collections.Counter(r["label"] for r in valid_records)
    per_class_splits = {
        c: dict(collections.Counter(r["split"] for r in valid_records if r["label"] == c))
        for c in classes
    }

    summary = {
        "collated_dir": str(args.collated),
        "classes": classes,
        "num_classes": len(classes),
        "feature_dim": FEATURE_DIM,
        "total_raw_files": int(sum(per_class_raw.values())),
        "exact_duplicates_skipped": int(exact_duplicates),
        "corrupted_files": corrupted_files,
        "unique_candidates": len(unique_records),
        "valid_samples": len(valid_records),
        "rejected_samples": int(sum(rejected_by_class.values())),
        "rejected_reasons": dict(rejected_reasons),
        "padding_fallback_rescued": int(padding_used_count),
        "near_duplicate_group_merges": int(merges),
        "split_counts": split_counts,
        "two_hand_detections_total": int(sum(two_hand_by_class.values())),
        "two_hand_by_class": dict(two_hand_by_class),
        "top_image_sizes": dict(image_sizes.most_common(15)),
        "per_class": {
            c: {
                "raw_files": int(per_class_raw[c]),
                "exact_duplicates_skipped": int(per_class_exact_dups[c]),
                "valid_samples": int(per_class_valid[c]),
                "rejected_samples": int(rejected_by_class[c]),
                "splits": per_class_splits[c],
            }
            for c in classes
        },
        "rejected_examples": rejected_samples,
    }
    summary_path = args.output / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Extraction complete: valid={len(valid_records)}, rejected={sum(rejected_by_class.values())}, "
        f"padding_rescued={padding_used_count}, merges={merges}, splits={split_counts}",
        flush=True,
    )
    print("Per-class valid/rejected:", {c: f"{per_class_valid[c]}/{rejected_by_class[c]}" for c in classes}, flush=True)


if __name__ == "__main__":
    main()
