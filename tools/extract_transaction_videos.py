"""Extract fresh MediaPipe hand and upper-body pose keypoints from raw
``coordinates/videos/<LABEL>/*.mp4`` recordings.

Mirrors the landmark layout and validation rules of ``tools/extract_fsl105.py``
(``(T, 75, 3)`` with 33 pose landmarks + 21 hand0 + 21 hand1), never reads or
uses the legacy ``coordinates/<LABEL>/*.npy`` files, and builds a leak-free
signer-disjoint train/val/test split using visual Ward clustering on frame-0
head/torso crops.

Outputs:
- ``transaction_keypoints/transaction_clips.npz``
- ``transaction_keypoints/dataset_summary.json``
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "coordinates" / "videos"
OUT_DIR = ROOT / "transaction_keypoints"
ARCHIVE_PATH = OUT_DIR / "transaction_clips.npz"
SUMMARY_PATH = OUT_DIR / "dataset_summary.json"
FSL105_LABELS_PATH = ROOT / "assets" / "ml" / "fsl105_labels.json"

DEFAULT_HAND_TASK = (
    Path.home()
    / "AppData"
    / "Local"
    / "Pub"
    / "Cache"
    / "hosted"
    / "pub.dev"
    / "hand_landmarker-2.2.0"
    / "android"
    / "src"
    / "main"
    / "assets"
    / "hand_landmarker.task"
)
DEFAULT_POSE_TASK = (
    ROOT / "android" / "app" / "src" / "main" / "assets" / "pose_landmarker_lite.task"
)

# Ward clusters verified visually on signer_ward22.jpg:
# each has exactly 52 clips (2 per class across all 26 classes) and zero overlap.
TEST_CLUSTERS = {3, 5, 7, 12}
VAL_CLUSTERS = {9, 10, 11}
MIN_USABLE_HAND_FRAMES = 4
INTER_CLIP_TS_GAP_MS = 10_000


def _resolve_task_paths() -> tuple[Path, Path]:
    hand_task = DEFAULT_HAND_TASK
    if not hand_task.exists():
        matches = list(
            (Path.home() / "AppData" / "Local" / "Pub" / "Cache").rglob(
                "hand_landmarker.task"
            )
        )
        if not matches:
            raise FileNotFoundError("Could not locate hand_landmarker.task")
        hand_task = matches[0]
    pose_task = DEFAULT_POSE_TASK
    if not pose_task.exists():
        raise FileNotFoundError(f"Could not locate {pose_task}")
    return hand_task, pose_task


def _clip_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def discover_videos() -> tuple[list[str], list[dict]]:
    classes = sorted(p.name for p in VIDEOS_DIR.iterdir() if p.is_dir())
    records: list[dict] = []
    for class_id, label in enumerate(classes):
        clips = sorted(
            (VIDEOS_DIR / label).glob("*.mp4"),
            key=_clip_index,
        )
        for path in clips:
            records.append(
                {
                    "class_id": class_id,
                    "label": label,
                    "clip_id": _clip_index(path),
                    "path": str(path),
                }
            )
    return classes, records


def assign_signer_disjoint_splits(records: list[dict]) -> None:
    """Cluster clips by frame-0 head/torso appearance to build leak-free splits."""
    vecs = []
    for rec in records:
        cap = cv2.VideoCapture(rec["path"])
        ok, fr0 = cap.read()
        cap.release()
        if not ok or fr0 is None:
            raise RuntimeError(f"Unreadable video during split assignment: {rec['path']}")
        h, w = fr0.shape[:2]
        center = cv2.resize(
            fr0[int(h * 0.12) : int(h * 0.88), int(w * 0.32) : int(w * 0.68)],
            (24, 32),
        ).astype(np.float32)
        bg = cv2.resize(fr0, (24, 18)).astype(np.float32)
        vecs.append(np.concatenate([center.reshape(-1) * 1.5, bg.reshape(-1) * 0.7]))

    x = np.stack(vecs, axis=0)
    n = len(x)
    sums = x.copy()
    counts = np.ones(n, dtype=np.float32)
    active = list(range(n))
    members = {i: [i] for i in range(n)}

    while len(active) > 22:
        idx = np.array(active)
        means = sums[idx] / counts[idx][:, None]
        norms = np.sum(means * means, axis=1)
        d2 = np.maximum(0.0, norms[:, None] + norms[None, :] - 2.0 * (means @ means.T))
        ward = (
            d2
            * (counts[idx][:, None] * counts[idx][None, :])
            / (counts[idx][:, None] + counts[idx][None, :] + 1e-12)
        )
        np.fill_diagonal(ward, np.inf)
        a_pos, b_pos = divmod(int(np.argmin(ward)), len(active))
        ia, ib = active[a_pos], active[b_pos]
        sums[ia] += sums[ib]
        counts[ia] += counts[ib]
        members[ia].extend(members[ib])
        del members[ib]
        active.pop(b_pos)

    clusters = sorted([members[i] for i in active], key=len, reverse=True)
    for cluster_id, clip_indices in enumerate(clusters):
        if cluster_id in TEST_CLUSTERS:
            split = "test"
        elif cluster_id in VAL_CLUSTERS:
            split = "val"
        else:
            split = "train"
        for i in clip_indices:
            records[i]["signer_cluster"] = cluster_id
            records[i]["split"] = split


def _extract_worker(args: tuple[int, list[dict], str, str]) -> list[dict]:
    worker_id, chunk, hand_task_path, pose_task_path = args
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_task_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_task_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    results: list[dict] = []
    ts_ms = 0
    with (
        vision.HandLandmarker.create_from_options(hand_opts) as hand_landmarker,
        vision.PoseLandmarker.create_from_options(pose_opts) as pose_landmarker,
    ):
        for rec in chunk:
            cap = cv2.VideoCapture(rec["path"])
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if fps <= 1.0:
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            aspect = float(width / height) if height > 0 else (4.0 / 3.0)
            step_ms = max(1, round(1000.0 / fps))

            frames_lm: list[np.ndarray] = []
            hand_frames = 0
            two_hand_frames = 0
            pose_frames = 0

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                h_res = hand_landmarker.detect_for_video(mp_img, ts_ms)
                p_res = pose_landmarker.detect_for_video(mp_img, ts_ms)
                ts_ms += step_ms

                row = np.zeros((75, 3), dtype=np.float32)
                if p_res.pose_landmarks:
                    pose_lms = p_res.pose_landmarks[0]
                    for j, lm in enumerate(pose_lms[:33]):
                        row[j, 0] = lm.x
                        row[j, 1] = lm.y
                        row[j, 2] = lm.z
                    pose_frames += 1

                if h_res.hand_landmarks:
                    hand_frames += 1
                    if len(h_res.hand_landmarks) >= 2:
                        two_hand_frames += 1
                    for slot, h_lms in enumerate(h_res.hand_landmarks[:2]):
                        base = 33 + slot * 21
                        for j, lm in enumerate(h_lms[:21]):
                            row[base + j, 0] = lm.x
                            row[base + j, 1] = lm.y
                            row[base + j, 2] = lm.z

                np.nan_to_num(row, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                frames_lm.append(row)

            cap.release()
            ts_ms += INTER_CLIP_TS_GAP_MS

            seq = (
                np.stack(frames_lm, axis=0)
                if frames_lm
                else np.zeros((0, 75, 3), dtype=np.float32)
            )
            results.append(
                {
                    **rec,
                    "seq": seq,
                    "aspect": aspect,
                    "fps": fps,
                    "hand_frames": hand_frames,
                    "two_hand_frames": two_hand_frames,
                    "pose_frames": pose_frames,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    t0 = time.time()
    hand_task, pose_task = _resolve_task_paths()
    classes, records = discover_videos()
    print(f"Discovered {len(records)} videos across {len(classes)} classes.", flush=True)

    assign_signer_disjoint_splits(records)
    split_dist = {
        s: sum(1 for r in records if r["split"] == s) for s in ("train", "val", "test")
    }
    print(f"Signer-disjoint split candidate counts: {split_dist}", flush=True)

    workers = max(1, min(args.workers, len(records)))
    indexed_chunks: list[list[tuple[int, dict]]] = [[] for _ in range(workers)]
    for i, rec in enumerate(records):
        indexed_chunks[i % workers].append((i, rec))

    extracted: list[dict | None] = [None] * len(records)
    print(
        f"Running fresh MediaPipe VIDEO extraction across {workers} workers...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                _extract_worker,
                (
                    wid,
                    [rec for _, rec in chunk],
                    str(hand_task),
                    str(pose_task),
                ),
            ): [orig_i for orig_i, _ in chunk]
            for wid, chunk in enumerate(indexed_chunks)
        }
        done = 0
        for fut in as_completed(future_map):
            orig_indices = future_map[fut]
            worker_res = fut.result()
            for orig_i, item in zip(orig_indices, worker_res):
                extracted[orig_i] = item
            done += len(worker_res)
            print(f"  completed {done}/{len(records)} videos ({time.time() - t0:.1f}s)", flush=True)

    usable: list[dict] = []
    dropped: list[dict] = []
    for item in extracted:
        assert item is not None
        t_len = int(item["seq"].shape[0])
        if t_len == 0 or item["hand_frames"] < MIN_USABLE_HAND_FRAMES:
            dropped.append(
                {
                    "class_id": item["class_id"],
                    "label": item["label"],
                    "clip_id": item["clip_id"],
                    "reason": f"too_few_hand_frames ({item['hand_frames']}/{t_len})",
                }
            )
            continue
        usable.append(item)

    if not usable:
        raise RuntimeError("No usable clips extracted.")

    offsets = np.zeros(len(usable) + 1, dtype=np.int64)
    pos = 0
    seqs = []
    for i, item in enumerate(usable):
        offsets[i] = pos
        seqs.append(item["seq"])
        pos += item["seq"].shape[0]
    offsets[len(usable)] = pos
    raw_landmarks = np.concatenate(seqs, axis=0).astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARCHIVE_PATH,
        raw_landmarks=raw_landmarks,
        clip_offsets=offsets,
        class_ids=np.array([item["class_id"] for item in usable], dtype=np.int32),
        clip_ids=np.array([item["clip_id"] for item in usable], dtype=np.int32),
        aspects=np.array([item["aspect"] for item in usable], dtype=np.float32),
        fps=np.array([item["fps"] for item in usable], dtype=np.float32),
        splits=np.array([item["split"] for item in usable], dtype="<U5"),
        signer_clusters=np.array(
            [item["signer_cluster"] for item in usable], dtype=np.int32
        ),
        labels=np.array(classes),
    )

    fsl105_set: set[str] = set()
    if FSL105_LABELS_PATH.exists():
        with open(FSL105_LABELS_PATH, "r", encoding="utf-8") as f:
            fsl105_set = {entry["key"] for entry in json.load(f)}

    class_summaries = []
    for cid, label in enumerate(classes):
        cls_items = [item for item in usable if item["class_id"] == cid]
        total_vids = sum(1 for r in records if r["class_id"] == cid)
        class_summaries.append(
            {
                "id": cid,
                "key": label,
                "in_fsl105": label in fsl105_set,
                "total_videos": total_vids,
                "usable_clips": len(cls_items),
                "splits": {
                    s: sum(1 for item in cls_items if item["split"] == s)
                    for s in ("train", "val", "test")
                },
                "two_hand_clips": sum(
                    1 for item in cls_items if item["two_hand_frames"] >= 4
                ),
                "mean_hand_frame_ratio": round(
                    float(
                        np.mean(
                            [item["hand_frames"] / max(1, item["seq"].shape[0]) for item in cls_items]
                        )
                    ),
                    4,
                ),
                "mean_pose_frame_ratio": round(
                    float(
                        np.mean(
                            [item["pose_frames"] / max(1, item["seq"].shape[0]) for item in cls_items]
                        )
                    ),
                    4,
                ),
            }
        )

    summary = {
        "source_videos_dir": str(VIDEOS_DIR.relative_to(ROOT)),
        "total_classes": len(classes),
        "overlap_with_fsl105": sorted(c for c in classes if c in fsl105_set),
        "new_classes": sorted(c for c in classes if c not in fsl105_set),
        "total_candidate_videos": len(records),
        "usable_clips": len(usable),
        "dropped_clips": dropped,
        "split_counts": {
            s: sum(1 for item in usable if item["split"] == s)
            for s in ("train", "val", "test")
        },
        "test_signer_clusters": sorted(TEST_CLUSTERS),
        "val_signer_clusters": sorted(VAL_CLUSTERS),
        "total_frames_packed": int(raw_landmarks.shape[0]),
        "clips_archive": str(ARCHIVE_PATH.relative_to(ROOT)),
        "elapsed_seconds": round(time.time() - t0, 2),
        "classes": class_summaries,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Done in {time.time() - t0:.1f}s! Usable={len(usable)}/{len(records)}, "
        f"dropped={len(dropped)}, saved {ARCHIVE_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
