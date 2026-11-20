"""Train paired 41-sign trials. Usage: python tools/train_body_trial.py FSL105_CACHE OUTPUT_DIR."""
import argparse
import csv
import json
from pathlib import Path

from train_continuous import NEW, digest
from train_signs import ROOT, HIDDEN, STEPS, features, forward, np, reflected, train

ADDED = {"IM_FINE", "NICE_TO_MEET_YOU", "SEE_YOU_TOMORROW", "MONDAY", "TUESDAY",
         "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"}
UNKNOWN = "__OTHER__"


def mirror(x):
    result = reflected(x)
    if result.shape[-1] >= 104:
        result[..., 91:103:2] *= -1
    return result


def encode(raw, aspect):
    # Pose and hands arrive upright. Use image height as the common XY unit.
    raw = raw.copy()
    raw[..., 0] *= aspect
    hands = raw[:, 33:].reshape(-1, 2, 21, 3)
    xy = hands[..., :2]
    visible = np.any(xy, axis=(2, 3))
    keep = visible.any(axis=1)
    if keep.sum() < 3:
        return None
    base = features(xy)
    order = np.argsort(np.where(visible, xy[:, :, 0, 1], np.inf), axis=1, kind="stable")
    hands = np.take_along_axis(hands, order[:, :, None, None], axis=1)[keep]
    visible = np.take_along_axis(visible, order, axis=1)[keep]
    pose = raw[keep, :33, :2]
    center = pose[:, [11, 12]].mean(axis=1)
    width = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=1)
    valid = (width > .025) & np.any(pose[:, [0, 11, 12, 13, 14]] != 0, axis=2).all(axis=1)
    body = np.zeros((len(pose), 13), dtype="f")
    depth = np.zeros((len(pose), 10), dtype="f")
    for slot in range(2):
        wrist = hands[:, slot, 0, :2]
        elbows = pose[:, [13, 14]]
        near = np.argmin(np.linalg.norm(elbows - wrist[:, None], axis=2), axis=1)
        elbow = elbows[np.arange(len(pose)), near]
        offsets = np.concatenate([wrist-center, elbow-center, wrist-pose[:, 0]], axis=1)
        body[:, slot*6:slot*6+6] = np.where((valid & visible[:, slot])[:, None],
                                            offsets / np.maximum(width[:, None], .025), 0)
        # MediaPipe Z is wrist-relative and measured in image-width units.
        palm = np.linalg.norm(hands[:, slot, [5, 9, 13, 17], :2] - wrist[:, None], axis=2).mean(axis=1)
        depth[:, slot*5:slot*5+5] = ((hands[:, slot, [4, 8, 12, 16, 20], 2] -
                                      hands[:, slot, 0:1, 2]) * aspect / np.maximum(palm[:, None], .0001))
    body[:, -1] = valid
    extra = np.clip(np.concatenate([body, depth], axis=1), -5, 5)
    t = np.linspace(0, len(extra)-1, STEPS)
    lo, hi = np.floor(t).astype(int), np.ceil(t).astype(int)
    return np.column_stack([base, extra[lo]+(extra[hi]-extra[lo])*(t-lo)[:, None]]).astype("f")


def recordings(cache, labels):
    for folder in sorted((ROOT / "coordinates").iterdir()):
        paths = sorted(folder.glob("*.npy"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
        for i, path in enumerate(paths):
            split = "test" if i >= len(paths)-round(.2*len(paths)) else "val" if i >= round(.7*len(paths)) else "train"
            yield folder.name, split, "transaction", np.load(path).reshape(-1, 75, 3), 4/3, 25
    with (cache / "metadata/labels.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = int(row["id"])
            label = row["label"].strip().replace(" ", "_")
            if label == "YOURE_WELCOME":  # Possible synonym of transactional WELCOME.
                continue
            folder, = (cache / "landmarks_v2").glob(f"{cid:03}_*")
            for video in sorted((ROOT / "videos" / str(cid)).glob("*.MOV")):
                clip = int(video.stem)
                # The fourth five-clip signer block (including extras) is held out.
                split = "train" if clip < 10 else "val" if clip < 15 else "test"
                if label not in labels:
                    # Negative labels are disjoint across train, validation, and test.
                    if ("train", "val", "test")[cid % 3] != split:
                        continue
                if digest(video) != digest(cache / "videos" / str(cid) / video.name):
                    raise ValueError(f"Video mismatch: {video}")
                with np.load(folder / f"{clip}.npz") as d:
                    if (int(d["class_id"]), int(d["clip_id"]), str(d["label"]).strip()) != (cid, clip, row["label"].strip()):
                        raise ValueError(f"Metadata mismatch: {video}")
                    yield label if label in labels else UNKNOWN, split, "FSL105", d["raw_landmarks"], float(d["aspect"]), float(d["fps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    labels = sorted(NEW | ADDED | {d.name for d in (ROOT/"coordinates").iterdir() if d.is_dir()}) + [UNKNOWN]
    rng = np.random.default_rng(2026)
    inputs, targets, splits, sources = [], [], [], []
    missed = []
    for label, split, source, raw, aspect, fps in recordings(args.cache, labels):
        for rate in ([15, 10, 7] if split == "train" else [12]):
            stride = max(1, round(fps/rate))
            x = encode(raw[(rng.integers(stride) if split == "train" else 0)::stride], aspect)
            if x is None:
                if split != "train":
                    missed.append((split, source, label))
                continue
            inputs.append(x)
            targets.append(labels.index(label))
            splits.append(split)
            sources.append(source)
    x, y, split, source = map(np.asarray, (inputs, targets, splits, sources))
    print("Usable views", dict(zip(*np.unique(split, return_counts=True))), "missed", missed, flush=True)
    other = len(labels)-1
    for name, size in [("hands", 91), ("body", 104), ("body_depth", 114)]:
        matrix = np.ascontiguousarray(x[:, :, :size])
        w = train(matrix[split=="train"], y[split=="train"], len(labels), epochs=48, mirror=mirror)
        def classify(part):
            v = matrix[split==part]
            logits = (forward(v, w)[0]+forward(mirror(v), w)[0])*.5
            e = np.exp(logits-logits.max(axis=1, keepdims=True))
            p = e/e.sum(axis=1, keepdims=True)
            return p.argmax(axis=1), p.max(axis=1)
        # Select the confidence cutoff on validation data only, weighting false
        # words and false accepts more heavily than a retry.
        pred, score = classify("val")
        true = y[split=="val"]
        def utility(t):
            accept = (pred != other) & (score >= t)
            known = true != other
            return ((accept & (pred==true))[known].mean() -
                    2*(accept & (pred!=true))[known].mean() - (accept[~known]).mean())
        threshold = float(max(np.arange(.60, 1., .01), key=utility))
        validation_utility = utility(threshold)
        model = dict(version=1 if size==91 else 2, features=size, hidden=HIDDEN, steps=STEPS,
                     labels=labels, threshold=threshold, **{k:a.flatten().round(7).tolist() for k,a in w.items()})
        (args.output/f"{name}.json").write_text(json.dumps(model,separators=(",",":")))
        for part in ("val", "test"):
            pred, score = classify(part)
            true = y[split==part]
            accept = (pred!=other) & (score>=threshold)
            groups = {s:(source[split==part]==s) & (true!=other) for s in ("transaction", "FSL105")}
            groups.update({l:true==labels.index(l) for l in sorted(ADDED | {"HELLO","FIVE",UNKNOWN})})
            for group, mask in groups.items():
                print(name, part, group, "clips", int(mask.sum()), "correct words", int((mask & accept & (true==pred)).sum()),
                      "wrong words", int((mask & accept & (true!=pred)).sum()), "retry", int((mask & ~accept).sum()))
        print(name, "validation utility", round(validation_utility,4), "threshold", threshold, flush=True)


if __name__ == "__main__":
    main()

