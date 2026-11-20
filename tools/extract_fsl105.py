"""Verify and extract MediaPipe hand + pose keypoints from the FSL-105 video dataset.

Preserves original FSL-105 label IDs (0..104) while excluding ID 2 (GOOD EVENING)
from the active 104-class taxonomy to prevent confusion with GOOD AFTERNOON (ID 1).

When a verified landmark extraction cache produced with the app's MediaPipe
task models is present and matches each video's SHA-256 hash, it reuses those
landmarks and packs them into `fsl105_keypoints/fsl105_clips.npz`. Any missing,
mismatched, or `--force`-requested videos are extracted directly with MediaPipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEOS = ROOT / "videos"
DEFAULT_OUT = ROOT / "fsl105_keypoints"
DEFAULT_CACHE = Path(r"C:\Users\Hans\Downloads\Visualin\Visualin-main\fsl105")
DEFAULT_POSE_TASK = ROOT / "android" / "app" / "src" / "main" / "assets" / "pose_landmarker_lite.task"
DEFAULT_HAND_TASK = Path(
    r"C:\Users\Hans\AppData\Local\Pub\Cache\hosted\pub.dev\hand_landmarker-2.2.0\android\src\main\assets\hand_landmarker.task"
)

EXCLUDED_CLASS_IDS = {2}

# Canonical FSL-105 table (original id, raw CSV label, display label, app key, category).
FSL105_TABLE: list[tuple[int, str, str, str, str]] = [
    (0, "GOOD MORNING", "GOOD MORNING", "GOOD_MORNING", "GREETING"),
    (1, "GOOD AFTERNOON", "GOOD AFTERNOON", "GOOD_AFTERNOON", "GREETING"),
    (2, "GOOD EVENING", "GOOD EVENING", "GOOD_EVENING", "GREETING"),
    (3, "HELLO", "HELLO", "HELLO", "GREETING"),
    (4, "HOW ARE YOU", "HOW ARE YOU", "HOW_ARE_YOU", "GREETING"),
    (5, "IM FINE", "IM FINE", "IM_FINE", "GREETING"),
    (6, "NICE TO MEET YOU", "NICE TO MEET YOU", "NICE_TO_MEET_YOU", "GREETING"),
    (7, "THANK YOU", "THANK YOU", "THANK_YOU", "GREETING"),
    (8, "YOURE WELCOME", "YOURE WELCOME", "YOURE_WELCOME", "GREETING"),
    (9, "SEE YOU TOMORROW", "SEE YOU TOMORROW", "SEE_YOU_TOMORROW", "GREETING"),
    (10, "UNDERSTAND", "UNDERSTAND", "UNDERSTAND", "SURVIVAL"),
    (11, "DON’T UNDERSTAND", "DON’T UNDERSTAND", "DONT_UNDERSTAND", "SURVIVAL"),
    (12, "KNOW", "KNOW", "KNOW", "SURVIVAL"),
    (13, "DON’T KNOW", "DON’T KNOW", "DONT_KNOW", "SURVIVAL"),
    (14, "NO", "NO", "NO", "SURVIVAL"),
    (15, "YES", "YES", "YES", "SURVIVAL"),
    (16, "WRONG", "WRONG", "WRONG", "SURVIVAL"),
    (17, "CORRECT", "CORRECT", "CORRECT", "SURVIVAL"),
    (18, "SLOW", "SLOW", "SLOW", "SURVIVAL"),
    (19, "FAST", "FAST", "FAST", "SURVIVAL"),
    (20, "ONE", "ONE", "ONE", "NUMBER"),
    (21, "TWO", "TWO", "TWO", "NUMBER"),
    (22, "THREE", "THREE", "THREE", "NUMBER"),
    (23, "FOUR", "FOUR", "FOUR", "NUMBER"),
    (24, "FIVE", "FIVE", "FIVE", "NUMBER"),
    (25, "SIX", "SIX", "SIX", "NUMBER"),
    (26, "SEVEN", "SEVEN", "SEVEN", "NUMBER"),
    (27, "EIGHT", "EIGHT", "EIGHT", "NUMBER"),
    (28, "NINE ", "NINE", "NINE", "NUMBER"),
    (29, "TEN", "TEN", "TEN", "NUMBER"),
    (30, "JANUARY", "JANUARY", "JANUARY", "CALENDAR"),
    (31, "FEBRUARY", "FEBRUARY", "FEBRUARY", "CALENDAR"),
    (32, "MARCH", "MARCH", "MARCH", "CALENDAR"),
    (33, "APRIL", "APRIL", "APRIL", "CALENDAR"),
    (34, "MAY", "MAY", "MAY", "CALENDAR"),
    (35, "JUNE", "JUNE", "JUNE", "CALENDAR"),
    (36, "JULY", "JULY", "JULY", "CALENDAR"),
    (37, "AUGUST", "AUGUST", "AUGUST", "CALENDAR"),
    (38, "SEPTEMBER", "SEPTEMBER", "SEPTEMBER", "CALENDAR"),
    (39, "OCTOBER", "OCTOBER", "OCTOBER", "CALENDAR"),
    (40, "NOVEMBER", "NOVEMBER", "NOVEMBER", "CALENDAR"),
    (41, "DECEMBER", "DECEMBER", "DECEMBER", "CALENDAR"),
    (42, "MONDAY", "MONDAY", "MONDAY", "DAYS"),
    (43, "TUESDAY", "TUESDAY", "TUESDAY", "DAYS"),
    (44, "WEDNESDAY", "WEDNESDAY", "WEDNESDAY", "DAYS"),
    (45, "THURSDAY", "THURSDAY", "THURSDAY", "DAYS"),
    (46, "FRIDAY", "FRIDAY", "FRIDAY", "DAYS"),
    (47, "SATURDAY", "SATURDAY", "SATURDAY", "DAYS"),
    (48, "SUNDAY", "SUNDAY", "SUNDAY", "DAYS"),
    (49, "TODAY", "TODAY", "TODAY", "DAYS"),
    (50, "TOMORROW", "TOMORROW", "TOMORROW", "DAYS"),
    (51, "YESTERDAY", "YESTERDAY", "YESTERDAY", "DAYS"),
    (52, "FATHER", "FATHER", "FATHER", "FAMILY"),
    (53, "MOTHER", "MOTHER", "MOTHER", "FAMILY"),
    (54, "SON", "SON", "SON", "FAMILY"),
    (55, "DAUGHTER", "DAUGHTER", "DAUGHTER", "FAMILY"),
    (56, "GRANDFATHER", "GRANDFATHER", "GRANDFATHER", "FAMILY"),
    (57, "GRANDMOTHER", "GRANDMOTHER", "GRANDMOTHER", "FAMILY"),
    (58, "UNCLE", "UNCLE", "UNCLE", "FAMILY"),
    (59, "AUNTIE", "AUNTIE", "AUNTIE", "FAMILY"),
    (60, "COUSIN", "COUSIN", "COUSIN", "FAMILY"),
    (61, "PARENTS", "PARENTS", "PARENTS", "FAMILY"),
    (62, "BOY", "BOY", "BOY", "RELATIONSHIPS"),
    (63, "GIRL", "GIRL", "GIRL", "RELATIONSHIPS"),
    (64, "MAN", "MAN", "MAN", "RELATIONSHIPS"),
    (65, "WOMAN", "WOMAN", "WOMAN", "RELATIONSHIPS"),
    (66, "DEAF", "DEAF", "DEAF", "RELATIONSHIPS"),
    (67, "HARD OF HEARING", "HARD OF HEARING", "HARD_OF_HEARING", "RELATIONSHIPS"),
    (68, "WEELCHAIR PERSON", "WHEELCHAIR PERSON", "WHEELCHAIR_PERSON", "RELATIONSHIPS"),
    (69, "BLIND", "BLIND", "BLIND", "RELATIONSHIPS"),
    (70, "DEAF BLIND", "DEAF BLIND", "DEAF_BLIND", "RELATIONSHIPS"),
    (71, "MARRIED", "MARRIED", "MARRIED", "RELATIONSHIPS"),
    (72, "BLUE", "BLUE", "BLUE", "COLOR"),
    (73, "GREEN", "GREEN", "GREEN", "COLOR"),
    (74, "RED", "RED", "RED", "COLOR"),
    (75, "BROWN", "BROWN", "BROWN", "COLOR"),
    (76, "BLACK", "BLACK", "BLACK", "COLOR"),
    (77, "WHITE", "WHITE", "WHITE", "COLOR"),
    (78, "YELLOW", "YELLOW", "YELLOW", "COLOR"),
    (79, "ORANGE", "ORANGE", "ORANGE", "COLOR"),
    (80, "GRAY", "GRAY", "GRAY", "COLOR"),
    (81, "PINK", "PINK", "PINK", "COLOR"),
    (82, "VIOLET", "VIOLET", "VIOLET", "COLOR"),
    (83, "LIGHT", "LIGHT", "LIGHT", "COLOR"),
    (84, "DARK", "DARK", "DARK", "COLOR"),
    (85, "BREAD", "BREAD", "BREAD", "FOOD"),
    (86, "EGG", "EGG", "EGG", "FOOD"),
    (87, "FISH", "FISH", "FISH", "FOOD"),
    (88, "MEAT", "MEAT", "MEAT", "FOOD"),
    (89, "CHICKEN", "CHICKEN", "CHICKEN", "FOOD"),
    (90, "SPAGHETTI", "SPAGHETTI", "SPAGHETTI", "FOOD"),
    (91, "RICE", "RICE", "RICE", "FOOD"),
    (92, "LONGANISA", "LONGANISA", "LONGANISA", "FOOD"),
    (93, "SHRIMP", "SHRIMP", "SHRIMP", "FOOD"),
    (94, "CRAB", "CRAB", "CRAB", "FOOD"),
    (95, "HOT", "HOT", "HOT", "DRINK"),
    (96, "COLD", "COLD", "COLD", "DRINK"),
    (97, "JUICE", "JUICE", "JUICE", "DRINK"),
    (98, "MILK", "MILK", "MILK", "DRINK"),
    (99, "COFFEE", "COFFEE", "COFFEE", "DRINK"),
    (100, "TEA", "TEA", "TEA", "DRINK"),
    (101, "BEER", "BEER", "BEER", "DRINK"),
    (102, "WINE", "WINE", "WINE", "DRINK"),
    (103, "SUGAR", "SUGAR", "SUGAR", "DRINK"),
    (104, "NO SUGAR", "NO SUGAR", "NO_SUGAR", "DRINK"),
]


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def split_for_clip(clip_id: int) -> str:
    # Recording order uses four blocks of 5 clips per signer (with extra clips >= 20 in block 4).
    # Signer 4 (clip >= 15) is strictly held out for final test evaluation.
    if clip_id >= 15:
        return "test"
    if clip_id % 5 == 4:
        return "val"
    return "train"


def extract_video_mediapipe(
    video: Path, hand_detector, pose_detector, target_fps: float = 30.0, ts_offset: int = 0
) -> tuple[np.ndarray, float, float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(source_fps / target_fps)))

    import mediapipe as mp

    rows: list[np.ndarray] = []
    source_index = -1
    last_ts = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_index += 1
            if source_index % step:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(round(source_index / source_fps * 1000.0))
            last_ts = ts
            detect_ts = ts_offset + ts
            hand_res = hand_detector.detect_for_video(image, detect_ts)
            pose_res = pose_detector.detect_for_video(image, detect_ts)

            row = np.zeros((75, 3), dtype=np.float32)
            if pose_res.pose_landmarks:
                for i, p in enumerate(pose_res.pose_landmarks[0][:33]):
                    row[i] = (p.x, p.y, p.z)
            for slot, h_lms in enumerate(hand_res.hand_landmarks[:2]):
                base = 33 + slot * 21
                for i, p in enumerate(h_lms[:21]):
                    row[base + i] = (p.x, p.y, p.z)
            rows.append(row)
    finally:
        cap.release()

    arr = np.stack(rows) if rows else np.zeros((0, 75, 3), dtype=np.float32)
    aspect = float(width / height) if height > 0 else 1.0
    out_fps = float(source_fps / step)
    return arr, aspect, out_fps, ts_offset + last_ts + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--hand-task", type=Path, default=DEFAULT_HAND_TASK)
    parser.add_argument("--pose-task", type=Path, default=DEFAULT_POSE_TASK)
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-extract with MediaPipe")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    active_rows = [row for row in FSL105_TABLE if row[0] not in EXCLUDED_CLASS_IDS]
    assert len(active_rows) == 104

    label_entries = []
    for model_idx, (orig_id, csv_label, label, key, category) in enumerate(active_rows):
        label_entries.append({
            "model_index": model_idx,
            "id": orig_id,
            "label": label,
            "csv_label": csv_label,
            "key": key,
            "category": category,
        })

    labels_asset = ROOT / "assets" / "ml" / "fsl105_labels.json"
    labels_asset.parent.mkdir(parents=True, exist_ok=True)
    labels_asset.write_text(json.dumps(label_entries, indent=2, ensure_ascii=False), encoding="utf-8")

    hand_detector = None
    pose_detector = None
    ts_offset = 0

    def ensure_detectors():
        nonlocal hand_detector, pose_detector
        if hand_detector is not None:
            return hand_detector, pose_detector
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(args.hand_task)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(args.pose_task)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        return hand_detector, pose_detector

    landmark_chunks: list[np.ndarray] = []
    clip_offsets: list[int] = [0]
    clip_class_ids: list[int] = []
    clip_model_indices: list[int] = []
    clip_ids: list[int] = []
    clip_aspects: list[float] = []
    clip_fps: list[float] = []
    clip_splits: list[str] = []

    dropped_clips: list[dict] = []
    class_summaries: list[dict] = []
    cache_reused = 0
    freshly_extracted = 0

    try:
        for entry in label_entries:
            cid = entry["id"]
            model_idx = entry["model_index"]
            vdir = args.videos / str(cid)
            videos = sorted(vdir.glob("*.MOV"), key=lambda p: int(p.stem))
            cache_lm_dir = next((args.cache / "landmarks_v2").glob(f"{cid:03d}_*"), None) if args.cache.exists() else None

            usable_count = 0
            two_hand_clips = 0
            pose_rates = []
            hand_rates = []

            for vid in videos:
                clip_id = int(vid.stem)
                cache_vid = args.cache / "videos" / str(cid) / vid.name if args.cache.exists() else None
                cache_npz = cache_lm_dir / f"{clip_id}.npz" if cache_lm_dir else None

                loaded = False
                if (
                    not args.force
                    and cache_vid is not None
                    and cache_vid.exists()
                    and cache_npz is not None
                    and cache_npz.exists()
                    and sha256_file(vid) == sha256_file(cache_vid)
                ):
                    with np.load(cache_npz) as d:
                        if int(d["class_id"]) == cid and int(d["clip_id"]) == clip_id:
                            raw = np.asarray(d["raw_landmarks"], dtype=np.float32)
                            aspect = float(d["aspect"])
                            fps = float(d["fps"])
                            loaded = True
                            cache_reused += 1

                if not loaded:
                    hd, pd = ensure_detectors()
                    raw, aspect, fps, ts_offset = extract_video_mediapipe(vid, hd, pd, ts_offset=ts_offset)
                    freshly_extracted += 1

                if raw.ndim != 3 or raw.shape[1:] != (75, 3) or not np.isfinite(raw).all():
                    dropped_clips.append({"class_id": cid, "label": entry["key"], "clip_id": clip_id, "reason": "invalid_shape_or_nan"})
                    continue

                h0 = np.any(raw[:, 33:54, :2] != 0, axis=(1, 2))
                h1 = np.any(raw[:, 54:75, :2] != 0, axis=(1, 2))
                any_hand = h0 | h1
                hand_frames = int(any_hand.sum())
                if hand_frames < 5:
                    dropped_clips.append({
                        "class_id": cid,
                        "label": entry["key"],
                        "clip_id": clip_id,
                        "reason": f"too_few_hand_frames ({hand_frames}/{len(raw)})",
                    })
                    continue

                usable_count += 1
                if int((h0 & h1).sum()) >= max(3, int(0.3 * hand_frames)):
                    two_hand_clips += 1
                pose_ok = float(np.all(np.any(raw[:, [0, 11, 12, 13, 14], :2] != 0, axis=2), axis=1).mean())
                pose_rates.append(pose_ok)
                hand_rates.append(float(hand_frames / max(1, len(raw))))

                landmark_chunks.append(raw)
                clip_offsets.append(clip_offsets[-1] + len(raw))
                clip_class_ids.append(cid)
                clip_model_indices.append(model_idx)
                clip_ids.append(clip_id)
                clip_aspects.append(aspect)
                clip_fps.append(fps)
                clip_splits.append(split_for_clip(clip_id))

            class_summaries.append({
                "model_index": model_idx,
                "id": cid,
                "key": entry["key"],
                "label": entry["label"],
                "category": entry["category"],
                "total_videos": len(videos),
                "usable_clips": usable_count,
                "two_hand_clips": two_hand_clips,
                "mean_hand_frame_ratio": round(float(np.mean(hand_rates)), 4) if hand_rates else 0.0,
                "mean_upper_body_pose_ratio": round(float(np.mean(pose_rates)), 4) if pose_rates else 0.0,
            })
    finally:
        if hand_detector is not None:
            hand_detector.close()
        if pose_detector is not None:
            pose_detector.close()

    all_landmarks = np.concatenate(landmark_chunks, axis=0).astype(np.float32)
    npz_path = args.out / "fsl105_clips.npz"
    np.savez_compressed(
        npz_path,
        raw_landmarks=all_landmarks,
        clip_offsets=np.asarray(clip_offsets, dtype=np.int64),
        class_ids=np.asarray(clip_class_ids, dtype=np.int32),
        model_indices=np.asarray(clip_model_indices, dtype=np.int32),
        clip_ids=np.asarray(clip_ids, dtype=np.int32),
        aspects=np.asarray(clip_aspects, dtype=np.float32),
        fps=np.asarray(clip_fps, dtype=np.float32),
        splits=np.asarray(clip_splits, dtype="<U5"),
    )

    split_counts = {s: clip_splits.count(s) for s in ("train", "val", "test")}
    summary = {
        "total_classes_in_csv": len(FSL105_TABLE),
        "excluded_class_ids": sorted(EXCLUDED_CLASS_IDS),
        "supported_classes": len(label_entries),
        "total_candidate_videos": sum(c["total_videos"] for c in class_summaries),
        "usable_clips": len(clip_ids),
        "dropped_clips": dropped_clips,
        "split_counts_fsl105": split_counts,
        "cache_reused_clips": cache_reused,
        "fresh_mediapipe_clips": freshly_extracted,
        "total_frames_packed": int(len(all_landmarks)),
        "clips_archive": str(npz_path.relative_to(ROOT)),
        "labels_asset": str(labels_asset.relative_to(ROOT)),
        "classes": class_summaries,
    }
    summary_path = args.out / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Extracted {len(clip_ids)}/{summary['total_candidate_videos']} usable clips "
        f"across {len(label_entries)} classes (excluded IDs {sorted(EXCLUDED_CLASS_IDS)}); "
        f"reused={cache_reused}, fresh={freshly_extracted}, dropped={len(dropped_clips)}"
    )
    print(f"Splits: {split_counts}")
    print(f"Wrote {npz_path} ({npz_path.stat().st_size / (1024*1024):.1f} MB) and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
