"""Train, evaluate, and export the static FSL alphabet classifier to TFLite."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from extract_alphabet import FEATURE_DIM, ROOT, compute_features_from_upright

DEFAULT_DATA = ROOT / "alphabet_keypoints" / "alphabet_features.npz"
DEFAULT_OUT_DIR = ROOT / "alphabet_keypoints"
DEFAULT_ASSETS_DIR = ROOT / "assets" / "ml"


def build_reflection_vector() -> np.ndarray:
    """Return length-88 vector that flips nx_0..nx_20 and keeps y, z, and geometric features."""
    r = np.ones(FEATURE_DIM, dtype=np.float32)
    r[0:63:3] = -1.0
    return r


def augment_landmarks(
    landmarks: np.ndarray,
    labels: np.ndarray,
    copies: int = 5,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply training-only phone-camera 3D landmark perturbations and recompute 88-dim features."""
    rng = np.random.default_rng(seed)
    feats_out = []
    y_out = []

    for pts, label in zip(landmarks, labels):
        orig_feats = compute_features_from_upright(pts)
        if orig_feats is not None:
            feats_out.append(orig_feats)
            y_out.append(int(label))

        wrist = pts[0]
        local = pts - wrist[None, :]
        palm = float(np.mean(np.linalg.norm(local[[5, 9, 13, 17], :2], axis=-1)))
        if palm <= 1e-4:
            continue

        for _ in range(copies):
            theta = float(
                np.clip(rng.normal(0.0, np.deg2rad(12.0)), -np.deg2rad(28.0), np.deg2rad(28.0))
            )
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            aug = local.copy()
            aug[:, 0] *= float(rng.uniform(0.75, 1.30))
            aug[:, :2] = aug[:, :2] @ rot.T
            if rng.random() < 0.25:
                aug[:, 2] = 0.0
            else:
                aug[:, 2] = aug[:, 2] * float(rng.uniform(0.15, 1.40)) + rng.normal(
                    0.0, 0.025 * palm, size=(21,)
                ).astype(np.float32)
            aug[:, :2] += rng.normal(0.0, 0.015 * palm, size=(21, 2)).astype(np.float32)
            f = compute_features_from_upright(aug + wrist[None, :])
            if f is not None:
                feats_out.append(f)
                y_out.append(int(label))

    return np.asarray(feats_out, dtype=np.float32), np.asarray(y_out, dtype=np.int32)


class MacroF1EarlyStopping(tf.keras.callbacks.Callback):
    """Stop training when validation macro-F1 peaks and restore best weights."""

    def __init__(self, x_val: np.ndarray, y_val: np.ndarray, patience: int = 12):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.patience = patience
        self.best_f1 = -1.0
        self.best_weights = None
        self.wait = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs=None):
        probs = self.model.predict(self.x_val, batch_size=256, verbose=0)
        preds = probs.argmax(axis=1)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            self.y_val, preds, average="macro", zero_division=0
        )
        macro_f1 = float(macro_f1)
        if logs is not None:
            logs["val_macro_f1"] = macro_f1
        if macro_f1 > self.best_f1 + 1e-5:
            self.best_f1 = macro_f1
            self.best_weights = self.model.get_weights()
            self.wait = 0
            self.best_epoch = epoch + 1
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_mirror_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    seed: int = 2026,
) -> tuple[tf.keras.Model, tf.keras.Model, dict]:
    """Train a mirror-symmetric MLP and fold it into a pure 3-layer sequential Dense MLP."""
    tf.keras.utils.set_random_seed(seed)
    r_vec = build_reflection_vector()

    inp = tf.keras.Input(shape=(FEATURE_DIM,), dtype=tf.float32, name="landmarks")
    ref_mat = np.diag(r_vec).astype(np.float32)
    ref_inp = tf.keras.layers.Dense(
        FEATURE_DIM,
        use_bias=False,
        trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(ref_mat),
        name="reflect_input",
    )(inp)

    h1_layer = tf.keras.layers.Dense(192, activation="relu", name="shared_fc1")
    h1_dir = h1_layer(inp)
    h1_ref = h1_layer(ref_inp)
    h1_sym = tf.keras.layers.Average(name="mirror_avg")([h1_dir, h1_ref])
    h1_drop = tf.keras.layers.Dropout(0.18)(h1_sym)
    h2 = tf.keras.layers.Dense(128, activation="relu", name="fc2")(h1_drop)
    h2_drop = tf.keras.layers.Dropout(0.10)(h2)
    probs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(h2_drop)

    train_model = tf.keras.Model(inp, probs, name="fsl_alphabet_mirror_train")

    steps_per_epoch = max(1, int(np.ceil(len(x_train) / 128)))
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2.5e-3,
        decay_steps=steps_per_epoch * 75,
        alpha=0.05,
    )
    train_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=2e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
        metrics=["accuracy"],
    )

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    early_stop = MacroF1EarlyStopping(x_val, y_val, patience=14)

    hist = train_model.fit(
        x_train,
        y_train_oh,
        validation_data=(x_val, y_val_oh),
        epochs=75,
        batch_size=128,
        verbose=2,
        callbacks=[early_stop],
    )

    # Fold the mirror average directly into a pure 3-layer sequential Dense MLP:
    # Layer 1: [A, diag(R) @ A], [c, c] (88 -> 384, relu)
    # Layer 2: [0.5 * B; 0.5 * B], d (384 -> 128, relu)
    # Layer 3: W_out, b_out (128 -> 26, softmax)
    a_w, c_b = train_model.get_layer("shared_fc1").get_weights()
    b_w, d_b = train_model.get_layer("fc2").get_weights()
    out_w, out_b = train_model.get_layer("probs").get_weights()

    w1 = np.concatenate([a_w, r_vec[:, None] * a_w], axis=1).astype(np.float32)
    b1 = np.concatenate([c_b, c_b], axis=0).astype(np.float32)
    w2 = np.concatenate([0.5 * b_w, 0.5 * b_w], axis=0).astype(np.float32)
    b2 = d_b.astype(np.float32)

    exp_in = tf.keras.Input(shape=(FEATURE_DIM,), dtype=tf.float32, name="landmarks")
    e1 = tf.keras.layers.Dense(384, activation="relu", name="fc1")(exp_in)
    e2 = tf.keras.layers.Dense(128, activation="relu", name="fc2")(e1)
    e3 = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(e2)
    export_model = tf.keras.Model(exp_in, e3, name="fsl_alphabet_mlp")
    export_model.get_layer("fc1").set_weights([w1, b1])
    export_model.get_layer("fc2").set_weights([w2, b2])
    export_model.get_layer("probs").set_weights([out_w, out_b])

    # Confirm exact equivalence between train_model (inference) and folded export_model
    p_train = train_model.predict(x_val, batch_size=256, verbose=0)
    p_export = export_model.predict(x_val, batch_size=256, verbose=0)
    fold_diff = float(np.max(np.abs(p_train - p_export)))
    if fold_diff > 1e-5:
        raise RuntimeError(f"Folded export model diverged from train model: {fold_diff}")

    history_summary = {
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": early_stop.best_epoch,
        "best_val_macro_f1": round(early_stop.best_f1, 6),
        "final_train_accuracy": round(float(hist.history["accuracy"][-1]), 6),
        "final_val_accuracy": round(float(hist.history["val_accuracy"][-1]), 6),
        "fold_max_abs_diff": fold_diff,
    }
    return train_model, export_model, history_summary


def train_baseline_63d_mlp(
    x_train_63: np.ndarray,
    y_train: np.ndarray,
    x_val_63: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    seed: int = 2026,
) -> tf.keras.Model:
    """Baseline MLP on 63 normalized coordinates only (for empirical comparison)."""
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(63,), dtype=tf.float32)
    h1 = tf.keras.layers.Dense(192, activation="relu")(inp)
    h1 = tf.keras.layers.Dropout(0.18)(h1)
    h2 = tf.keras.layers.Dense(128, activation="relu")(h1)
    h2 = tf.keras.layers.Dropout(0.10)(h2)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(h2)
    m = tf.keras.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    es = MacroF1EarlyStopping(x_val_63, y_val, patience=10)
    m.fit(
        x_train_63,
        y_train,
        validation_data=(x_val_63, y_val),
        epochs=50,
        batch_size=128,
        verbose=0,
        callbacks=[es],
    )
    return m


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    confused = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused.append(
                    {"true": classes[i], "predicted": classes[j], "count": int(cm[i, j])}
                )
    confused.sort(key=lambda item: item["count"], reverse=True)

    return {
        "accuracy": round(acc, 6),
        "macro_precision": round(float(p_mac), 6),
        "macro_recall": round(float(r_mac), 6),
        "macro_f1": round(float(f_mac), 6),
        "weighted_f1": round(float(f_w), 6),
        "per_class": {
            c: {
                "precision": round(float(report[c]["precision"]), 6),
                "recall": round(float(report[c]["recall"]), 6),
                "f1": round(float(report[c]["f1-score"]), 6),
                "support": int(report[c]["support"]),
            }
            for c in classes
        },
        "top_confusions": confused[:20],
        "confusion_matrix": cm.tolist(),
    }


def calibrate_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    best_t = 0.70
    best_util = -1e9
    best_stats = {}
    for t in np.arange(0.50, 0.955, 0.02):
        accept = confs >= t
        correct_acc = float((accept & (preds == y_true)).mean())
        wrong_acc = float((accept & (preds != y_true)).mean())
        reject_rate = float((~accept).mean())
        util = correct_acc - 2.0 * wrong_acc - 0.15 * reject_rate
        if util > best_util:
            best_util = util
            best_t = round(float(t), 2)
            best_stats = {
                "threshold": best_t,
                "utility": round(util, 6),
                "accept_rate": round(float(accept.mean()), 6),
                "accepted_accuracy": round(float((preds[accept] == y_true[accept]).mean()) if accept.any() else 0.0, 6),
                "wrong_accept_rate": round(wrong_acc, 6),
                "reject_rate": round(reject_rate, 6),
            }
    return best_t, best_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.data, allow_pickle=False) as d:
        x_raw = d["X"].astype(np.float32)
        landmarks = d["landmarks"].astype(np.float32)
        y = d["y"].astype(np.int32)
        classes = [str(c) for c in d["classes"]]
        split = np.asarray([str(s) for s in d["split"]])

    num_classes = len(classes)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    x_val, y_val = x_raw[val_mask], y[val_mask]
    x_test, y_test = x_raw[test_mask], y[test_mask]
    landmarks_test = landmarks[test_mask]

    print(
        f"Loaded dataset: train={int(train_mask.sum())}, val={int(val_mask.sum())}, test={int(test_mask.sum())}",
        flush=True,
    )

    x_train_aug, y_train_aug = augment_landmarks(landmarks[train_mask], y[train_mask], copies=5, seed=2026)
    print(f"Augmented train samples: {len(x_train_aug)}", flush=True)

    # 1. Baseline comparisons
    baseline_63 = train_baseline_63d_mlp(x_train_aug[:, :63], y_train_aug, x_val[:, :63], y_val, num_classes)
    b63_val_pred = baseline_63.predict(x_val[:, :63], batch_size=256, verbose=0).argmax(axis=1)
    b63_test_pred = baseline_63.predict(x_test[:, :63], batch_size=256, verbose=0).argmax(axis=1)

    rf = RandomForestClassifier(n_estimators=250, random_state=2026, n_jobs=-1)
    rf.fit(x_train_aug, y_train_aug)
    rf_val_pred = rf.predict(x_val)
    rf_test_pred = rf.predict(x_test)

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=12.0, gamma="scale"))
    svm.fit(x_train_aug, y_train_aug)
    svm_val_pred = svm.predict(x_val)
    svm_test_pred = svm.predict(x_test)

    # 2. Primary Mirror-Symmetric 88-dim MLP
    _, export_model, train_history = train_mirror_mlp(
        x_train_aug, y_train_aug, x_val, y_val, num_classes, seed=2026
    )

    keras_path = args.out_dir / "alphabet_model.keras"
    export_model.save(keras_path)

    # 3. Export to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    tflite_bytes = converter.convert()
    tflite_path = args.assets_dir / "alphabet_classifier.tflite"
    tflite_tmp = tflite_path.with_suffix(".tmp")
    tflite_tmp.write_bytes(tflite_bytes)
    tflite_tmp.replace(tflite_path)

    # 4. Verify TFLite vs Keras and exact mirror invariance
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    def run_tflite(batch: np.ndarray) -> np.ndarray:
        out = np.empty((len(batch), num_classes), dtype=np.float32)
        for idx in range(len(batch)):
            interp.set_tensor(in_det["index"], batch[idx : idx + 1].astype(np.float32))
            interp.invoke()
            out[idx] = interp.get_tensor(out_det["index"])[0]
        return out

    keras_val_probs = export_model.predict(x_val, batch_size=256, verbose=0)
    keras_test_probs = export_model.predict(x_test, batch_size=256, verbose=0)
    tflite_val_probs = run_tflite(x_val)
    tflite_test_probs = run_tflite(x_test)

    r_vec = build_reflection_vector()
    tflite_test_reflected = run_tflite(x_test * r_vec[None, :])

    tflite_max_diff = float(np.max(np.abs(keras_test_probs - tflite_test_probs)))
    tflite_label_agree = float((keras_test_probs.argmax(axis=1) == tflite_test_probs.argmax(axis=1)).mean())
    mirror_max_diff = float(np.max(np.abs(tflite_test_probs - tflite_test_reflected)))

    print(
        f"TFLite verification: size={len(tflite_bytes)} bytes, max_abs_diff={tflite_max_diff:.2e}, "
        f"label_agree={tflite_label_agree:.4f}, mirror_max_diff={mirror_max_diff:.2e}",
        flush=True,
    )
    if tflite_max_diff > 1e-4 or tflite_label_agree < 1.0 or mirror_max_diff > 1e-5:
        raise RuntimeError("TFLite verification failed tolerance checks")

    calibrated_threshold, val_thresh_stats = calibrate_threshold(tflite_val_probs, y_val)
    # Cap runtime threshold for live phone camera tolerance while keeping false accepts bounded.
    threshold = round(min(calibrated_threshold, 0.52), 2)
    val_metrics = evaluate_predictions(y_val, tflite_val_probs.argmax(axis=1), classes)
    test_metrics = evaluate_predictions(y_test, tflite_test_probs.argmax(axis=1), classes)

    # Thresholded test performance
    test_preds = tflite_test_probs.argmax(axis=1)
    test_confs = tflite_test_probs.max(axis=1)
    test_accept = test_confs >= threshold
    test_thresh_stats = {
        "threshold": threshold,
        "accept_rate": round(float(test_accept.mean()), 6),
        "accepted_accuracy": round(float((test_preds[test_accept] == y_test[test_accept]).mean()) if test_accept.any() else 0.0, 6),
        "overall_accepted_correct_rate": round(float((test_accept & (test_preds == y_test)).mean()), 6),
        "wrong_accept_rate": round(float((test_accept & (test_preds != y_test)).mean()), 6),
    }

    # Comparison summary
    comparisons = {
        "mlp_63d_baseline": {
            "val_accuracy": round(float(accuracy_score(y_val, b63_val_pred)), 6),
            "val_macro_f1": round(float(precision_recall_fscore_support(y_val, b63_val_pred, average="macro", zero_division=0)[2]), 6),
            "test_accuracy": round(float(accuracy_score(y_test, b63_test_pred)), 6),
            "test_macro_f1": round(float(precision_recall_fscore_support(y_test, b63_test_pred, average="macro", zero_division=0)[2]), 6),
        },
        "random_forest_88d": {
            "val_accuracy": round(float(accuracy_score(y_val, rf_val_pred)), 6),
            "val_macro_f1": round(float(precision_recall_fscore_support(y_val, rf_val_pred, average="macro", zero_division=0)[2]), 6),
            "test_accuracy": round(float(accuracy_score(y_test, rf_test_pred)), 6),
            "test_macro_f1": round(float(precision_recall_fscore_support(y_test, rf_test_pred, average="macro", zero_division=0)[2]), 6),
        },
        "svm_rbf_88d": {
            "val_accuracy": round(float(accuracy_score(y_val, svm_val_pred)), 6),
            "val_macro_f1": round(float(precision_recall_fscore_support(y_val, svm_val_pred, average="macro", zero_division=0)[2]), 6),
            "test_accuracy": round(float(accuracy_score(y_test, svm_test_pred)), 6),
            "test_macro_f1": round(float(precision_recall_fscore_support(y_test, svm_test_pred, average="macro", zero_division=0)[2]), 6),
        },
        "mirror_mlp_88d_tflite": {
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        },
    }

    # Save confusion matrix CSV (test set)
    cm_path = args.out_dir / "confusion_matrix.csv"
    with cm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + classes)
        for label, row in zip(classes, test_metrics["confusion_matrix"]):
            writer.writerow([label] + row)

    eval_payload = {
        "train_samples_raw": int(train_mask.sum()),
        "train_samples_augmented": int(len(x_train_aug)),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "training": train_history,
        "tflite": {
            "path": str(tflite_path.relative_to(ROOT)),
            "bytes": len(tflite_bytes),
            "input_shape": [int(v) for v in in_det["shape"]],
            "input_dtype": "float32",
            "output_shape": [int(v) for v in out_det["shape"]],
            "output_dtype": "float32",
            "keras_vs_tflite_max_abs_diff": tflite_max_diff,
            "keras_vs_tflite_label_agreement": tflite_label_agree,
            "mirror_invariance_max_abs_diff": mirror_max_diff,
        },
        "model_comparisons": comparisons,
        "threshold_calibration_val": val_thresh_stats,
        "thresholded_test": test_thresh_stats,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (args.out_dir / "evaluation_metrics.json").write_text(
        json.dumps(eval_payload, indent=2), encoding="utf-8"
    )

    # Build 26 held-out test samples (one correctly classified per letter where available, or highest-conf sample)
    verification_samples = []
    for cls_idx, label in enumerate(classes):
        idxs = np.flatnonzero(y_test == cls_idx)
        correct_idxs = [int(i) for i in idxs if tflite_test_probs[i].argmax() == cls_idx]
        chosen = correct_idxs[0] if correct_idxs else int(idxs[0])
        verification_samples.append(
            {
                "label": label,
                "landmarks": landmarks_test[chosen].round(6).tolist(),
                "features": x_test[chosen].round(6).tolist(),
                "expectedTopLabel": classes[int(tflite_test_probs[chosen].argmax())],
                "expectedTopScore": round(float(tflite_test_probs[chosen].max()), 6),
                "expectedProbs": [round(float(v), 6) for v in tflite_test_probs[chosen]],
            }
        )

    preprocessing_config = {
        "version": 1,
        "modelAsset": "assets/ml/alphabet_classifier.tflite",
        "inputShape": [1, FEATURE_DIM],
        "dtype": "float32",
        "labels": classes,
        "threshold": threshold,
        "stabilityFrames": 3,
        "windowMs": 1500,
        "minHoldMs": 180,
        "disappearClearMs": 450,
        "dynamicLetters": ["J", "Z"],
        "normalization": {
            "coordinateSpace": "upright_image_height_units",
            "origin": "wrist_landmark_0",
            "scaleReference": "mean_2d_distance_wrist_to_mcp_5_9_13_17",
            "clampRange": [-5.0, 5.0],
            "featureLayout": {
                "0_to_62": "21x3 wrist-relative palm-normalized (x, y, z)",
                "63_to_72": "10 2D XY Euclidean distances from thumb tip (4) to [5,6,8,9,10,12,13,14,17,20]",
                "73_to_77": "5 2D XY Euclidean distances from wrist (0) to fingertips [4,8,12,16,20]",
                "78_to_81": "4 2D XY Euclidean distances between adjacent tips [(8,12),(12,16),(16,20),(4,16)]",
                "82_to_86": "5 2D XY finger curl cosines [(1,2,2,4),(5,6,6,8),(9,10,10,12),(13,14,14,16),(17,18,18,20)]",
                "87": "2D XY crossover cosine between (8-12) and (5-9) for R vs U",
            },
        },
        "testSamples": verification_samples,
    }
    prep_path = args.assets_dir / "alphabet_preprocessing.json"
    prep_tmp = prep_path.with_suffix(".tmp")
    prep_tmp.write_text(json.dumps(preprocessing_config, separators=(",", ":")), encoding="utf-8")
    prep_tmp.replace(prep_path)

    print("Model comparisons:", json.dumps(comparisons, indent=2), flush=True)
    print("Top test confusions:", test_metrics["top_confusions"][:10], flush=True)
    print(f"Threshold={threshold}, test_accept={test_thresh_stats}", flush=True)


if __name__ == "__main__":
    main()
