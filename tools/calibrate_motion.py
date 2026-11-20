"""Calibrate a per-label minimum motion on training clips and check it on validation clips.
Run against the evaluated body GRU. Numeric labels remain eligible as held handshapes.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from train_body_trial import UNKNOWN, ROOT, encode, forward, mirror, np, recordings

NUMBERS = set("ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN".split())


def motion(sequence):
    # Use the primary hand on a stable run of one-/two-hand frames. Quantiles
    # suppress an isolated jump. Compare body-relative wrist paths and finger
    # shape/depth; whole-person shifts should not look like signing.
    scores = []
    for two in (False, True):
        rows = sequence[(sequence[:, 88] > .95) if two else (sequence[:, 88] < .05)]
        if len(rows) < 6:
            continue
        for slot in range(2 if two else 1):
            selected = rows[:, slot*42:slot*42+42].reshape(-1,21,2)[:,[4,8,12,16,20]]
            depth = rows[:,104+slot*5:109+slot*5,None]
            tips = np.concatenate([selected,depth],axis=2)
            spans = np.quantile(tips,.8,axis=0)-np.quantile(tips,.2,axis=0)
            scores.append(np.sqrt((spans**2).sum(axis=1)).mean())
            body = rows[rows[:,103] > .95,91+slot*6:93+slot*6]
            if len(body) >= 6:
                extent = np.quantile(body,.8,axis=0)-np.quantile(body,.2,axis=0)
                scores.append(np.linalg.norm(extent))
    return max(scores,default=0.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache",type=Path)
    parser.add_argument("output", type=Path)
    args=parser.parse_args()
    model=json.loads((ROOT/"assets/ml/body_signs.json").read_text())
    labels=model["labels"]
    n=len(labels)
    weights={k:np.array(model[k],dtype="f").reshape(shape) for k,shape in
             dict(w=(114,192),u=(64,192),b=(192,),out=(178,n),bias=n).items()}
    by_source=defaultdict(list)
    rows=defaultdict(list)
    for label,split,source,raw,aspect,fps in recordings(args.cache,labels):
        raw=raw[::max(1,round(fps/12))]
        sequence=encode(raw,aspect)
        if sequence is None:
            continue
        energy=motion(sequence)
        if split=="train":
            by_source[label,source].append(energy)
            continue
        rows[split].append((label,energy,sequence))
    # Require consistent movement across the available training sources.
    # Use half the lower decile as a conservative floor above small jitter.
    floors={}
    for label in labels:
        if label in NUMBERS or label==UNKNOWN:
            continue
        groups=[v for (l,_),v in by_source.items() if l==label]
        floor=min(np.quantile(v,.10) for v in groups)*.5
        if floor>=.075:
            floors[label]=round(float(min(.30,floor)),5)
    for split in ("val","test"):
        data=rows[split]
        seq=np.stack([r[2] for r in data])
        logits=(forward(seq,weights)[0]+forward(mirror(seq),weights)[0])*.5
        e=np.exp(logits-logits.max(axis=1,keepdims=True));prob=e/e.sum(axis=1,keepdims=True)
        pred=np.where(prob.max(1)>=model["threshold"],prob.argmax(1),n-1)
        # Validation can only remove a gate if it drops any correct full clip.
        if split=="val":
            for (label,energy,_),idx in zip(data,pred):
                if labels[idx]==label and energy < floors.get(label,0):
                    floors.pop(label,None)
        removed=[(label,labels[idx]) for (label,energy,_),idx in zip(data,pred)
                 if energy < floors.get(labels[idx],0)]
        print(split,"withheld",removed)
    print("gates",floors)
    model["motionFloors"]=floors
    args.output.write_text(json.dumps(model,separators=(",",":")))


if __name__=="__main__":
    main()
