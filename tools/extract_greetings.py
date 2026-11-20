"""Extract the seven FSL-105 greetings with phone-style per-frame MediaPipe inference."""
import argparse
import hashlib
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GREETINGS = {0: "GOOD_MORNING", 1: "GOOD_AFTERNOON", 4: "HOW_ARE_YOU", 5: "IM_FINE",
             6: "NICE_TO_MEET_YOU", 8: "YOURE_WELCOME", 9: "SEE_YOU_TOMORROW"}


def extract(cid, hand, pose, output):
    label = GREETINGS[cid]
    folder = output / label
    folder.mkdir(exist_ok=True, parents=True)
    for video in sorted((ROOT / "videos" / str(cid)).glob("*.MOV")):
        dest = folder / (video.stem + ".npz")
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        if dest.exists():
            with np.load(dest) as d:
                if str(d["sha256"]) == digest:
                    continue
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            raise ValueError(video)
        step = max(1, round(fps/12))
        rows = []
        i = -1
        try:
            while cap.grab():
                i += 1
                if i % step:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(video)
                height, width = frame.shape[:2]
                factor = min(1,640/max(width,height))
                frame = cv2.resize(frame, (round(width*factor),round(height*factor)), interpolation=cv2.INTER_AREA)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                row = np.zeros((75,3),dtype="f")
                for slot, points in enumerate(hand.detect(image).hand_landmarks):
                    row[33+slot*21:54+slot*21] = [(p.x,p.y,p.z) for p in points]
                poses = pose.detect(image).pose_landmarks
                if poses:
                    row[:33] = [(p.x,p.y,p.z) for p in poses[0]]
                rows.append(row)
        finally:
            cap.release()
        if not rows:
            raise RuntimeError(video)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, raw_landmarks=rows, aspect=width/height, fps=fps/step,
                            sha256=digest, class_id=cid, clip_id=int(video.stem), label=label)
        tmp.replace(dest)
    print(label, len(list(folder.glob("*.npz"))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("hand_task",type=Path)
    parser.add_argument("ids",type=int,nargs="*",default=list(GREETINGS))
    args=parser.parse_args()
    cv2.setNumThreads(1)
    vision=mp.tasks.vision
    base=mp.tasks.BaseOptions
    h=vision.HandLandmarkerOptions(base_options=base(model_asset_path=str(args.hand_task)),num_hands=2,min_hand_detection_confidence=.7)
    p=vision.PoseLandmarkerOptions(base_options=base(model_asset_path=str(ROOT/"android/app/src/main/assets/pose_landmarker_lite.task")))
    with vision.HandLandmarker.create_from_options(h) as hand, vision.PoseLandmarker.create_from_options(p) as pose:
        for cid in args.ids:
            extract(cid,hand,pose,args.output)


if __name__=="__main__":
    main()
