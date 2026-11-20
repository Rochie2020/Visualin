"""Train the optional pause-separated model from transactional and FSL-105 clips.
Pass the FSL-105 extraction root (videos, metadata and landmarks_v2). Each
cached extraction is matched to this project's videos by class, clip and hash.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from train_signs import ROOT, STEPS, FEATURES, HIDDEN, features, forward, reflected, train, np

# Fixed before evaluation. The other FSL-105 classes stay outside this trial.
NEW = {"GOOD_MORNING", "GOOD_AFTERNOON", "HOW_ARE_YOU", "TODAY", "TOMORROW"}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").digest()


def clips(cache, labels):
    for folder in sorted((ROOT / "coordinates").iterdir()):
        files = sorted(folder.glob("*.npy"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
        boundary = len(files) - max(1, round(len(files) * .2))
        for i, path in enumerate(files):
            hands = np.load(path)[:, 99:].reshape(-1, 2, 21, 3)[..., :2].copy()
            hands[..., 0] *= 4 / 3
            yield "original", folder.name, i >= boundary, hands, 25
    with (cache / "metadata/test.csv").open(encoding="utf-8-sig") as src:
        held = {(int(r["id_label"]), int(Path(r["vid_path"].replace("\\", "/")).stem))
                for r in csv.DictReader(src)}
    with (cache / "metadata/labels.csv").open(encoding="utf-8-sig") as src:
        for row in csv.DictReader(src):
            label = row["label"].strip().replace(" ", "_")
            if label not in labels:
                continue
            cid = int(row["id"])
            directory, = (cache / "landmarks_v2").glob(f"{cid:03}_*")
            for video in sorted((ROOT / "videos" / str(cid)).glob("*.MOV")):
                if digest(video) != digest(cache / "videos" / str(cid) / video.name):
                    raise ValueError(f"Video differs from cached source: {video}")
                with np.load(directory / f"{video.stem}.npz") as data:
                    key = (cid, int(video.stem))
                    if (int(data["class_id"]), int(data["clip_id"])) != key or str(data["label"]).strip() != row["label"].strip():
                        raise ValueError(f"Label mismatch: {video}")
                    hands = data["raw_landmarks"][:, 33:, :2].reshape(-1, 2, 21, 2).copy()
                    hands[..., 0] *= float(data["aspect"])
                    yield "FSL-105", label, key in held, hands, float(data["fps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    labels = sorted(NEW | {d.name for d in (ROOT / "coordinates").iterdir() if d.is_dir()})
    x, y, test, sources = [], [], [], []
    rng = np.random.default_rng(2026)
    counts = {}
    for source, label, held, hands, fps in clips(args.cache, labels):
        counts[source] = counts.get(source, 0) + 1
        for rate in ([12] if held else [15, 12, 8, 6]):
            stride = max(1, round(fps / rate))
            sequence = hands[(0 if held else rng.integers(stride))::stride]
            if np.any(sequence, axis=(1, 2, 3)).sum() < 3:
                if held:
                    raise ValueError(f"No hands in held-out {source} {label}")
                continue
            x.append(features(sequence))
            y.append(labels.index(label))
            test.append(held)
            sources.append(source)
    x, y, test, sources = map(np.asarray, (x, y, test, sources))
    print("Clips", counts, flush=True)
    weights = train(x[~test], y[~test], len(labels))
    scores = (forward(x[test], weights)[0] + forward(reflected(x[test]), weights)[0]) / 2
    correct = scores.argmax(axis=1) == y[test]
    for source in counts:
        m = sources[test] == source
        print(source, int(correct[m].sum()), "/", int(m.sum()))
    for label in sorted(NEW):
        mask = y[test] == labels.index(label)
        print(label, int(correct[mask].sum()), "/", int(mask.sum()))
    model = dict(version=1, labels=labels, threshold=.80, steps=STEPS, features=FEATURES,
                 hidden=HIDDEN, **{k: a.flatten().round(7).tolist() for k, a in weights.items()})
    path = ROOT / "assets/ml/continuous_signs.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model, separators=(",", ":")))
    tmp.replace(path)


if __name__ == "__main__":
    main()

