"""Fit a greeting specialist alongside the 41-sign model. The last FSL signer
block is held out. For the seven targeted classes each of the three earlier
blocks contributes four training clips and one validation clip."""
import argparse
import hashlib
import json
from pathlib import Path

from train_body_trial import UNKNOWN, ROOT, encode, mirror, forward, np, recordings, train
from calibrate_motion import motion

IDS = {0:"GOOD_MORNING",1:"GOOD_AFTERNOON",4:"HOW_ARE_YOU",5:"IM_FINE",
       6:"NICE_TO_MEET_YOU",8:"YOURE_WELCOME",9:"SEE_YOU_TOMORROW"}
LABELS = list(IDS.values()) + [UNKNOWN]


def greeting_clips(cache, fresh):
    for cid,label in IDS.items():
        old, = (cache/"landmarks_v2").glob(f"{cid:03}_*")
        for video in sorted((ROOT/"videos"/str(cid)).glob("*.MOV")):
            clip = int(video.stem)
            split = "test" if clip>=15 else "val" if clip%5==4 else "train"
            path = fresh/label/f"{clip}.npz"
            with np.load(path) as d:
                if (int(d["class_id"]), int(d["clip_id"]),str(d["label"]),str(d["sha256"])) != (
                        cid,clip,label,hashlib.sha256(video.read_bytes()).hexdigest()):
                    raise ValueError(f"Invalid fresh extraction: {video}")
                yield label,split,d["raw_landmarks"],float(d["aspect"]),float(d["fps"])
            if split=="train":
                with np.load(old/f"{clip}.npz") as d:
                    yield label,split,d["raw_landmarks"],float(d["aspect"]),float(d["fps"])


def scores(x, weights):
    logits=(forward(x,weights)[0]+forward(mirror(x),weights)[0])*.5
    prob=np.exp(logits-logits.max(1,keepdims=True))
    return prob/prob.sum(1,keepdims=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    base=json.loads((ROOT/"assets/ml/body_signs.json").read_text())
    rng=np.random.default_rng(4107)
    x,y,parts,truth=[],[],[],[]
    tests=[]
    def add(label,split,raw,aspect,fps,target):
        views=[(1,12)] if split!="train" else ([(1,12),(1,8),(1,6),(2,12),(2,8)] if target else [(1,12)])
        for aug,rate in views:
            stride=max(1,round(fps/rate))
            frames=raw[(int(rng.integers(stride)) if split=="train" else 0)::stride].copy()
            if aug==2 and len(frames)>6:
                keep=rng.random(len(frames))>.15
                frames=frames[keep]
            features=encode(frames,aspect)
            if features is None:
                if split!="train": raise ValueError(f"Unusable held-out clip {label}")
                continue
            x.append(features);y.append(LABELS.index(label) if target else 7);parts.append(split);truth.append(label)
            if split=="test": tests.append((label,frames,aspect,round(1000*stride/fps)))
    for label,split,raw,aspect,fps in greeting_clips(a.cache,ROOT/"greeting_keypoints"):
        add(label,split,raw,aspect,fps,True)
    for label,split,source,raw,aspect,fps in recordings(a.cache,base["labels"]):
        if label not in IDS.values():
            add(label,split,raw,aspect,fps,False)
    x,y,parts,truth=map(np.asarray,(x,y,parts,truth))
    training=np.flatnonzero(parts=="train")
    # Balance greetings against the many unrelated clips. All augmented views
    # of a recording remain on the same side of the split.
    positives=training[y[training]!=7]
    indices=np.concatenate([training,positives,positives])
    w=train(x[indices],y[indices],8,seed=4107,epochs=60,mirror=mirror)
    val=parts=="val"
    pv=scores(x[val],w)
    winner=pv.argmax(1);conf=pv.max(1)
    thresholds=np.arange(.55,.995,.01)
    def metric(t):
        ok=(winner!=7)&(conf>=t)
        pos=y[val]!=7
        return ((ok & (winner==y[val]))[pos].mean()-
                2*(ok & (winner!=y[val]))[pos].mean()-
                4*ok[~pos].mean())
    threshold=float(max(thresholds,key=metric))
    spec=dict(version=2,features=114,steps=12,hidden=64,labels=LABELS,threshold=threshold,
              motionFloors={l:v for l,v in base.get("motionFloors",{}).items() if l in IDS.values()},
              **{k:v.flatten().round(7).tolist() for k,v in w.items()})
    a.output.mkdir(exist_ok=True,parents=True)
    (a.output/"greetings.json").write_text(json.dumps(spec,separators=(",",":")))
    pt=scores(x[parts=="test"],w)
    with (a.output/"test.ndjson").open("w") as out:
        for (label,frames,aspect,ms),prob in zip(tests,pt):
            encoded=[]
            for r in frames:
                h=[hand.round(7).tolist() for hand in r[33:].reshape(2,21,3) if np.any(hand[:,:2])]
                encoded.append([h,r[[0,11,12,13,14],:2].round(7).tolist()])
            print(json.dumps([label,float(aspect),ms,encoded],separators=(",",":")),file=out)
    print("Threshold",threshold)
    for l in LABELS:
        mask=y[parts=="test"]==LABELS.index(l)
        pr=pt.argmax(1);active=(pr!=7)&(pt.max(1)>=threshold)
        print(l,mask.sum(),int((mask&active&(pr==LABELS.index(l))).sum()),int((mask&active).sum()))


if __name__=="__main__":
    main()
