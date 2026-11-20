"""Train the bundled hand-only GRU. Run from the project root with Python and NumPy."""
import argparse
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

STEPS, FEATURES, HIDDEN = 12, 91, 64
ROOT = Path(__file__).resolve().parents[1]


def features(frames):
    """XY in common image-height units; zero hands indicate detector dropout."""
    exists = np.any(frames != 0, axis=(2, 3))
    order = np.argsort(np.where(exists, frames[:, :, 0, 1], np.inf), axis=1, kind="stable")
    hands = np.take_along_axis(frames, order[:, :, None, None], axis=1)
    present = np.take_along_axis(exists, order, axis=1)
    hands, present = hands[present[:, 0]], present[present[:, 0]]
    if len(hands) < 3:
        raise ValueError("Too few detected frames")
    wrist = hands[:, :, 0, :]
    local = hands - wrist[:, :, None, :]
    palm = np.mean(np.linalg.norm(local[:, :, [5, 9, 13, 17]], axis=-1), axis=2)
    primary_scale = max(float(np.median(palm[:, 0])), 0.0001)
    shape = local / np.maximum(palm[:, :, None, None], 0.0001)
    movement = (wrist[:, 0] - wrist[0, 0]) / primary_scale
    separation = (wrist[:, 1] - wrist[:, 0]) * present[:, 1, None] / primary_scale
    data = np.concatenate([shape.reshape(len(hands), 84), movement, separation,
                           present[:, 1, None], np.linalg.norm(movement, axis=1)[:, None],
                           np.linalg.norm(separation, axis=1)[:, None]], axis=1)
    data = np.clip(data, -5, 5)
    t = np.linspace(0, len(data) - 1, STEPS)
    lo, hi = np.floor(t).astype(int), np.ceil(t).astype(int)
    return (data[lo] + (data[hi] - data[lo]) * (t-lo)[:, None]).astype("f")


def reflected(x):
    x = x.copy()
    x[..., :88:2] *= -1
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def forward(x, p):
    h = np.zeros((len(x), HIDDEN), dtype="f")
    projected = x @ p["w"] + p["b"]
    history = []
    for i in range(STEPS):
        u = h @ p["u"]
        z, r = np.split(sigmoid(projected[:, i, :2*HIDDEN] + u[:, :2*HIDDEN]), 2, axis=1)
        n = np.tanh(projected[:, i, 2*HIDDEN:] + r * u[:, 2*HIDDEN:])
        history.append((h, z, r, n, u[:, 2*HIDDEN:]))
        h = z * h + (1-z) * n
    summary = np.concatenate([h, x.mean(axis=1)], axis=1)
    return summary @ p["out"] + p["bias"], (summary, history)


def train(x, y, classes, seed=37, epochs=65, mirror=reflected):
    rng = np.random.default_rng(seed)
    def weight(a, b):
        return (rng.normal(0, np.sqrt(2/(a+b)), (a,b))).astype("f")
    p = dict(w=weight(x.shape[-1], 3*HIDDEN), u=weight(HIDDEN, 3*HIDDEN),
             b=np.zeros(3*HIDDEN, "f"), out=weight(HIDDEN+x.shape[-1], classes),
             bias=np.zeros(classes, "f"))
    m = {k:np.zeros_like(v) for k,v in p.items()}
    v = {k:np.zeros_like(t) for k,t in p.items()}
    step = 0
    for epoch in range(epochs):
        indices = rng.permutation(len(x))
        for start in range(0,len(x),64):
            batch = indices[start:start+64]
            a = x[batch].copy()
            a[::2] = mirror(a[::2])
            logits, (summary, history) = forward(a, p)
            prob = np.exp(logits - logits.max(axis=1, keepdims=True))
            d = prob / prob.sum(axis=1, keepdims=True)
            d[np.arange(len(d)),y[batch]] -= 1
            d /= len(d)
            g = {k:np.zeros_like(t) for k,t in p.items()}
            g["out"], g["bias"] = summary.T @ d, d.sum(axis=0)
            dh = (d @ p["out"].T)[:, :HIDDEN]
            for i in range(STEPS-1, -1, -1):
                prev, z, r, n, un = history[i]
                dn = dh * (1-z) * (1-n*n)
                dz, dr = dh * (prev-n) * z*(1-z), dn*un*r*(1-r)
                dx = np.concatenate([dz, dr, dn], axis=1)
                du = np.concatenate([dz, dr, dn*r], axis=1)
                g["w"] += a[:,i].T @ dx
                g["b"] += dx.sum(axis=0)
                g["u"] += prev.T @ du
                dh = dh*z + du @ p["u"].T
            step += 1
            for k in p:
                m[k] = .9*m[k] + .1*g[k]
                v[k] = .999*v[k] + .001*g[k]**2
                lr = .002 * (1 if epoch < epochs*.75 else .35)
                p[k] -= lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=65)
    args = parser.parse_args()
    labels = sorted(p.name for p in (ROOT/"coordinates").iterdir() if p.is_dir())
    rng = np.random.default_rng(2026)
    inputs, outcomes, mirrored, test = [], [], [], []
    for label_id, label in enumerate(labels):
        files = sorted((ROOT/"coordinates"/label).glob("*.npy"),
                       key=lambda p:int(p.stem.rsplit("_",1)[1]))
        # Hold back the last consecutive block per sign before making any augmented views.
        cutoff = len(files) - max(1, round(.2*len(files)))
        for i, file in enumerate(files):
            arr = np.load(file, allow_pickle=False)
            if arr.shape != (75,225) or not np.isfinite(arr).all():
                raise ValueError(str(file))
            # 33 pose landmarks then 21 landmarks on each hand; videos are 640x480.
            hands = arr[:,99:].reshape(-1,2,21,3)[...,:2].copy()
            hands[...,0] *= 4/3
            # Downsample as the Android CPU detector typically sees fewer than 25 fps.
            variants = [hands[::2]]
            if i < cutoff:
                for stride in (3,4,5):
                    variants.append(hands[rng.integers(stride)::stride])
            for variant in variants:
                if np.any(variant, axis=(1,2,3)).sum() < 3:
                    continue
                inputs.append(features(variant))
                outcomes.append(label_id)
                test.append(i>=cutoff)
                if i>=cutoff:
                    mirror = variant[:, ::-1].copy()
                    present = np.any(mirror!=0,axis=(2,3))
                    mirror[...,0] = np.where(present[:,:,None],4/3-mirror[...,0],0)
                    mirrored.append(features(mirror))
    x,y,mask = np.array(inputs),np.array(outcomes),np.array(test)
    p = train(x[~mask], y[~mask], len(labels), epochs=args.epochs)
    def scores(a):
        logits = (forward(a,p)[0]+forward(reflected(a),p)[0])*.5
        q = np.exp(logits-logits.max(axis=1,keepdims=True))
        return q/q.sum(axis=1,keepdims=True)
    prob, mp = scores(x[mask]), scores(np.array(mirrored))
    assert np.allclose(prob,mp,atol=2e-5)
    correct = prob.argmax(axis=1)==y[mask]
    accept = prob.max(axis=1)>=.75
    print("Holdout", int(correct.sum()), "/", len(correct), "mirrored", int((mp.argmax(1)==y[mask]).sum()),
          "accepted", int(accept.sum()), "accepted correct", int((accept & correct).sum()), flush=True)
    print({l:f"{correct[y[mask]==i].sum()}/{(y[mask]==i).sum()}" for i,l in enumerate(labels)}, flush=True)
    # Deploy the actual evaluated weights. Held-out recordings stay outside training.
    model = dict(version=1, labels=labels, steps=STEPS, features=FEATURES, hidden=HIDDEN,
                 threshold=.75, **{k:v.flatten().round(7).tolist() for k,v in p.items()})
    dest = ROOT/"assets"/"ml"/"transaction_signs.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(model,separators=(",",":")),encoding="utf-8")
    tmp.replace(dest)


if __name__ == "__main__":
    main()

