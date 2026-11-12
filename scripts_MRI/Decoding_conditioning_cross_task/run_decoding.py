#!/usr/bin/env python3
"""Train the manuscript odor decoder on passive odors and test conditioning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

MRI_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MRI_SCRIPTS))
from Decoding_probe_cross_task import run_roi_crosstask_odor2preconditioned as decoder
from Decoding_probe_cross_task import aggregate_roi_crosstask_odor2preconditioned as aggregate
from rsa_conditioning_odor_identity import run_rsa as conditioning

WINDOWS = conditioning.WINDOWS
CLUSTERS = conditioning.CLUSTERS
CONDITIONING_RUNS = (1, 2, 3)
PASSIVE_RUNS = (1, 2)
SVM_C = 1.0
DEFAULT_EXCLUSIONS = {
    "sub-19": "manuscript_sample_exclusion",
    "sub-31": "manuscript_sample_exclusion",
    "sub-33": "manuscript_sample_exclusion",
    "sub-26": "sparse_correct_conditioning_trials",
    "sub-38": "conditioning_OFC_coverage_loss",
}


def physical_cue_labels(samples: pd.DataFrame, sub: str,
                        original_events_dir: Path) -> tuple[np.ndarray, str]:
    """Use the probe decoder's physical-odor convention, never fit label flips."""
    if sub != decoder.SUB14:
        if not samples["odor_num"].eq(samples["behavior_odor_num"]).all():
            raise ValueError("Unexpected beta/behavior odor-label swap; physical labels need verification.")
        return samples["odor_num"].to_numpy(dtype=int) - 1, "conditioning_beta_index"

    originals = []
    for run in sorted(samples["run"].unique()):
        path = original_events_dir / f"{sub}_task-odorpred_run-{run}_events_orig.tsv"
        if not path.is_file():
            raise FileNotFoundError(f"Original sub-14 conditioning events required: {path}")
        events = pd.read_csv(path, sep="\t")
        conditioning.require_columns(events, ["trial_type", "run", "trial", "cue_idx", "odor_num"], path)
        events = events.loc[events["trial_type"].eq("cue_stim"), ["run", "trial", "cue_idx", "odor_num"]]
        events = conditioning.integer_columns(events, ["run", "trial", "cue_idx", "odor_num"], path)
        if events.empty or not events["run"].eq(run).all():
            raise ValueError(f"Empty or inconsistent original conditioning events: {path}")
        originals.append(events.rename(columns={"cue_idx": "original_cue_idx", "odor_num": "physical_odor_num"}))
    originals = pd.concat(originals, ignore_index=True)
    if originals.duplicated(["run", "trial"]).any():
        raise ValueError("Duplicate run/trial keys in original sub-14 conditioning events.")
    ordered = samples[["run", "trial", "cue_idx", "behavior_odor_num"]].copy()
    ordered["_sample_order"] = np.arange(len(ordered))
    matched = ordered.merge(originals, on=["run", "trial"], how="left", validate="one_to_one").sort_values("_sample_order")
    if matched["physical_odor_num"].isna().any():
        raise ValueError("Some sub-14 conditioning trials lack original physical-odor labels.")
    if not matched["cue_idx"].eq(matched["original_cue_idx"]).all():
        raise ValueError("Original sub-14 conditioning cue identities disagree with selected betas.")
    physical = matched["physical_odor_num"].astype(int)
    if not physical.isin([1, 2]).all():
        raise ValueError("Original sub-14 food-odor labels must be 1 or 2.")
    # The original conditioning TSVs retain the behavioral convention, before
    # the additional GLM-event swap. Passive TSV generation also swaps sub-14's
    # design labels; the original conditioning convention matches that training
    # convention, as in the manuscript's original-probe-event label handling.
    if not physical.eq(matched["behavior_odor_num"]).all():
        raise ValueError("Sub-14 original conditioning/behavior label conventions disagree; verify physical odors.")
    return physical.to_numpy() - 1, "sub14_original_conditioning_events"


def load_matched_cues(base: Path, behavior_dir: Path, sub: str, windows,
                      original_events_dir: Path):
    frames, quality = conditioning.matched_samples(
        base, behavior_dir, sub, windows, CONDITIONING_RUNS,
    )
    reason = conditioning.eligibility(next(iter(frames.values())), CONDITIONING_RUNS, primary=False)
    if reason:
        return frames, quality, reason
    for window, frame in frames.items():
        labels, source = physical_cue_labels(frame, sub, original_events_dir)
        frame["physical_odor_num"] = labels + 1
        frame["true_label"] = labels
        frame["label_source"] = source
        quality[window]["cue_label_source"] = source
    reference = next(iter(frames.values()))["true_label"].to_numpy()
    if any(not np.array_equal(frame["true_label"].to_numpy(), reference) for frame in frames.values()):
        raise ValueError("Physical conditioning labels disagree between cue windows.")
    return frames, quality, ""


def prepare_classification(passive_raw: np.ndarray, passive_runs: np.ndarray,
                           cue_raw: dict[str, np.ndarray], cue_runs: dict[str, np.ndarray]):
    """Use one finite voxel set and training normalization across cue windows."""
    windows = list(cue_raw)
    if any(values.shape[1] != passive_raw.shape[1] for values in cue_raw.values()):
        raise ValueError("Passive and conditioning matrices have different ROI voxel counts.")
    combined = np.vstack([cue_raw[window] for window in windows])
    passive_used, cue_used, finite = decoder._drop_nonfinite_voxels(passive_raw, combined)
    passive_patterns = decoder._runwise_zscore(passive_used, passive_runs)
    normalized, offset = {}, 0
    for window in windows:
        stop = offset + len(cue_raw[window])
        normalized[window] = decoder._runwise_zscore(cue_used[offset:stop], cue_runs[window])
        offset = stop
    return passive_patterns, normalized, finite


def analyze_roi(base: Path, sub: str, frames: dict, quality: dict, roi_info,
                spec, mask_path: Path, output: Path):
    roi_img, roi_mask, *_ = roi_info
    voxel_indices = np.column_stack(np.where(roi_mask))
    passive_raw, passive_y, passive_runs, passive_meta, passive_affine = decoder._load_passive_odor_patterns(
        base, sub, PASSIVE_RUNS, voxel_indices,
    )
    first_passive = nib.load(str(passive_meta.iloc[0]["beta_path"]))
    if first_passive.shape != roi_img.shape or not np.allclose(passive_affine, roi_img.affine, atol=1e-5):
        raise ValueError("Passive images and conditioning ROI mask are not on the same grid.")
    counts = pd.crosstab(passive_runs, passive_y).reindex(index=PASSIVE_RUNS, columns=[0, 1], fill_value=0)
    if not np.all(counts.to_numpy() == 10):
        raise ValueError("Expected 10 trials per food odor in each of the two passive runs (40 total).")
    raw = {window: conditioning.roi._load_roi_patterns(frame.to_dict("records"), roi_img, roi_mask)
           for window, frame in frames.items()}
    run_ids = {window: frame["run"].to_numpy() for window, frame in frames.items()}
    passive_patterns, cue_patterns, finite = prepare_classification(passive_raw, passive_runs, raw, run_ids)
    passive_metrics, passive_pred, passive_distance = decoder._passive_leave_one_run_out(
        passive_patterns, passive_y, passive_runs, SVM_C,
    )
    model = decoder.SVC(kernel="linear", C=SVM_C)
    model.fit(passive_patterns, passive_y)
    n_raw, n_used = len(finite), int(finite.sum())
    loss = 1 - n_used / n_raw
    rows = {}
    for window, frame in frames.items():
        labels = frame["true_label"].to_numpy(dtype=int)
        predictions = model.predict(cue_patterns[window])
        distances = decoder._decision_distance(model, cue_patterns[window])
        metrics = decoder._binary_metrics(labels, predictions, distances)
        row = {
            "sub": sub, "phase": "conditioning", "window": window, "roi": spec.roi_name,
            "hemisphere": "bilateral",
            "cue_subset": "conditioned_odor_only", "cue_label_source": frame["label_source"].iloc[0],
            "roi_mask": str(mask_path), "svm_c": SVM_C,
            "normalization": "within_task_window_run_voxelwise_zscore",
            "finite_voxel_scope": "passive_and_all_selected_conditioning_windows",
            "n_voxels_raw": n_raw, "n_voxels_used": n_used,
            "n_voxels_dropped_nonfinite": n_raw - n_used,
            "voxel_loss_fraction": loss, "high_voxel_loss_20pct": loss >= 0.20,
            "n_passive_samples": len(passive_y), "n_cue_samples": len(labels),
            "n_cue_odor1": int((labels == 0).sum()), "n_cue_odor2": int((labels == 1).sum()),
            "n_cue_runs": int(frame["run"].nunique()),
            "cue_runs": ",".join(map(str, sorted(frame["run"].unique()))),
            "incomplete_requested_runs": set(frame["run"]) != set(CONDITIONING_RUNS),
            **quality[window],
            **{f"cue_{key}": value for key, value in metrics.items()},
            **{f"passive_cv_{key}": value for key, value in passive_metrics.items()},
        }
        destination = output / window / spec.key / "predictions" / sub
        destination.mkdir(parents=True, exist_ok=True)
        trial_predictions = frame.copy()
        trial_predictions.insert(0, "sub", sub)
        trial_predictions["window"] = window
        trial_predictions["predicted_label"] = predictions
        trial_predictions["decision_distance"] = distances
        trial_predictions["signed_distance"] = (2 * labels - 1) * distances
        trial_predictions["prediction_correct"] = predictions == labels
        trial_predictions.to_csv(destination / "conditioning_predictions.csv", index=False)
        training_predictions = passive_meta.copy()
        training_predictions.insert(0, "sub", sub)
        training_predictions["cv_predicted_label"] = passive_pred
        training_predictions["cv_decision_distance"] = passive_distance
        training_predictions.to_csv(destination / "passive_cv_predictions.csv", index=False)
        rows[window] = row
    return rows


def plot_results(by_cluster: dict, path: Path, window: str):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_conditioning_decoder_matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_conditioning_decoder_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(MRI_SCRIPTS.parent))
    from figure_style import configure_matplotlib, style_axis

    configure_matplotlib()
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    for i, (cluster, frame) in enumerate(by_cluster.items()):
        values = frame["cue_balanced_accuracy"].to_numpy() * 100
        offsets = np.linspace(-0.09, 0.09, len(values))
        color = "#7B5EA7" if cluster == "cluster1" else "#168C8C"
        axis.scatter(i + offsets, values, color=color, s=30, zorder=3)
        mean = values.mean()
        ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
        axis.errorbar(i, mean, yerr=ci, fmt="D", mfc="white", mec="#29343D", ecolor="#29343D", capsize=5, zorder=5)
        axis.text(i, -0.18, f"n = {len(values)}", ha="center", transform=axis.get_xaxis_transform(), fontsize=11)
    axis.axhline(50, linestyle="--", linewidth=1, color="#7A858D")
    axis.set_xticks(range(len(by_cluster)), ["Medial OFC" if key == "cluster1" else "Lateral OFC" for key in by_cluster])
    axis.set_ylabel("Cross-task balanced accuracy (%)")
    timing = "0–2 s" if window == "cueEARLY" else "0–5 s"
    axis.set_title(f"Odor localizer → conditioning\n{window} ({timing}); all available runs", fontsize=13)
    axis.set_xlim(-0.5, len(by_cluster) - 0.5)
    style_axis(axis)
    fig.text(0.98, 0.015, "Diamonds: mean ± 95% CI; dashed line: chance", ha="right", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_dir", type=Path, required=True)
    p.add_argument("--project_root", type=Path, default=MRI_SCRIPTS.parent)
    p.add_argument("--behavior_dir", type=Path)
    p.add_argument("--mask_dir", type=Path)
    p.add_argument("--output_dir", type=Path)
    p.add_argument("--plot_dir", type=Path)
    p.add_argument("--sub14_original_events_dir", type=Path)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--windows", nargs="+", choices=WINDOWS, default=list(WINDOWS))
    p.add_argument("--clusters", nargs="+", choices=CLUSTERS, default=list(CLUSTERS))
    p.add_argument("--skip_subs", nargs="*", default=list(DEFAULT_EXCLUSIONS))
    p.add_argument("--skip_plots", action="store_true")
    p.add_argument("--strict", action="store_true", help="Stop immediately on an unexpected failure.")
    p.add_argument("--dry_run", action="store_true", help="Print settings without reading MRI data or writing results.")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    project = args.project_root.expanduser().resolve()
    base = args.base_dir.expanduser().resolve()
    behavior = (args.behavior_dir or project / "beh_processed_subs").expanduser().resolve()
    masks_dir = (args.mask_dir or project / "masks" / "OFC_clusterMap_K2_gm0p1").expanduser().resolve()
    output = (args.output_dir or project / "roi_decoding" / "conditioning_crosstask_odor").expanduser().resolve()
    plots = (args.plot_dir or project / "plots" / "conditioning_crosstask_odor").expanduser().resolve()
    originals = (args.sub14_original_events_dir or base / "nii" / "sub-14" / "func_PAID" / "odorpred").expanduser().resolve()
    config = {
        "phase": "conditioning", "base_dir": str(base), "behavior_dir": str(behavior),
        "mask_dir": str(masks_dir), "output_dir": str(output), "plot_dir": str(plots),
        "sub14_original_events_dir": str(originals),
        "windows": {window: WINDOWS[window] for window in args.windows},
        "clusters": args.clusters, "skip_subs": args.skip_subs, "subjects": args.subjects,
        "passive_runs": PASSIVE_RUNS, "conditioning_runs": CONDITIONING_RUNS,
        "svm_c": SVM_C, "primary_metric": "cue_balanced_accuracy", "chance": 0.5,
        "hemisphere": "bilateral", "group_test": "one_sided_t_vs_chance_uncorrected",
        "focus": "cueEXT5s, lateral OFC; EARLY and medial OFC are comparisons",
        "normalization": "within_task_window_run_voxelwise_zscore",
        "finite_voxel_scope": "passive_and_all_selected_conditioning_windows",
        "trial_selection": "correct responded food trials, matched across selected windows",
        "known_scan_exclusion": "sub-35 run 3", "skip_plots": args.skip_plots,
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    config["run_status"] = "running"
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    try:
        for window in args.windows:
            folder = base / "GLM" / WINDOWS[window]
            if not folder.is_dir():
                raise FileNotFoundError(f"Conditioning GLM folder not found: {folder}")
        gm = base / "masks" / "gm_0.1.nii"
        if not gm.is_file():
            raise FileNotFoundError(f"Gray-matter mask not found: {gm}")
        roi_masks = {key: conditioning.roi._load_mask(masks_dir / CLUSTERS[key].mask_filename, gm)
                     for key in args.clusters}
        subjects = sorted(set(args.subjects)) if args.subjects else sorted({
            path.name for window in args.windows
            for path in (base / "GLM" / WINDOWS[window]).iterdir()
            if path.is_dir() and path.name.startswith("sub-")
        })
        if not subjects:
            raise RuntimeError("No conditioning subjects found.")
    except Exception as exc:
        config.update(run_status="failed", failures=[str(exc)])
        manifest.write_text(json.dumps(config, indent=2) + "\n")
        raise
    config["discovered_subjects"] = subjects
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    bundles = {(window, cluster): {"rows": [], "qc": []} for window in args.windows for cluster in args.clusters}
    failures = []

    def record_all(sub, status, reason, quality=None):
        for (window, cluster), bundle in bundles.items():
            bundle["qc"].append({"sub": sub, "window": window, "roi": CLUSTERS[cluster].roi_name,
                                 "status": status, "reason": reason, **(quality or {}).get(window, {})})

    try:
        for sub in subjects:
            if sub in args.skip_subs:
                record_all(sub, "excluded", DEFAULT_EXCLUSIONS.get(sub, "explicit_skip_subs_list"))
                continue
            try:
                frames, quality, reason = load_matched_cues(base, behavior, sub, args.windows, originals)
                if reason:
                    record_all(sub, "ineligible", reason, quality)
                    continue
            except Exception as exc:
                record_all(sub, "failed", str(exc))
                failures.append(f"{sub}: {exc}")
                if args.strict:
                    raise
                print(f"[FAIL] {failures[-1]}", flush=True)
                continue
            for cluster in args.clusters:
                spec = CLUSTERS[cluster]
                try:
                    rows = analyze_roi(base, sub, frames, quality, roi_masks[cluster], spec,
                                       masks_dir / spec.mask_filename, output)
                    for window, row in rows.items():
                        bundles[window, cluster]["rows"].append(row)
                        bundles[window, cluster]["qc"].append({**row, "status": "ok", "reason": ""})
                        print(f"[OK] {sub} {window} {cluster}: "
                              f"balanced accuracy={row['cue_balanced_accuracy']:.4f}; "
                              f"trials={row['n_cue_samples']}; voxels={row['n_voxels_used']}/{row['n_voxels_raw']}", flush=True)
                except Exception as exc:
                    for window in args.windows:
                        bundles[window, cluster]["qc"].append({"sub": sub, "window": window,
                            "roi": spec.roi_name, "status": "failed", "reason": str(exc), **quality[window]})
                    failures.append(f"{sub} {cluster}: {exc}")
                    if args.strict:
                        raise
                    print(f"[FAIL] {failures[-1]}", flush=True)
    finally:
        # Persist completed rows and QC even when --strict stops partway through.
        for (window, cluster), bundle in bundles.items():
            destination = output / window / CLUSTERS[cluster].key
            destination.mkdir(parents=True, exist_ok=True)
            results = pd.DataFrame(bundle["rows"])
            if results.empty:
                results = pd.DataFrame(columns=["sub", "phase", "window", "roi", "cue_subset", *aggregate.METRIC_NULLS])
            results.to_csv(destination / "subject_results.csv", index=False)
            pd.DataFrame(bundle["qc"], columns=list(dict.fromkeys(
                ["sub", "window", "roi", "status", "reason"] + [k for row in bundle["qc"] for k in row]
            ))).to_csv(destination / "qc.csv", index=False)
            summary = aggregate._summarize(results)
            if not summary.empty:
                summary.insert(0, "window", window)
                summary.insert(1, "roi", CLUSTERS[cluster].roi_name)
                summary.insert(2, "phase", "conditioning")
            else:
                summary = pd.DataFrame(columns=["window", "roi", "phase", "metric", "n", "mean", "p_greater_than_null"])
            summary.to_csv(destination / "group_summary.csv", index=False)
        config["run_status"] = "failed" if failures or sys.exc_info()[0] is not None else "running"
        config["failures"] = failures
        manifest.write_text(json.dumps(config, indent=2) + "\n")
    for window in args.windows:
        populated = {cluster: pd.DataFrame(bundles[window, cluster]["rows"]) for cluster in args.clusters
                     if bundles[window, cluster]["rows"]}
        if len(populated) != len(args.clusters):
            failures.append(f"No valid decoding results for one or more ROIs: {window}")
        if populated and not args.skip_plots:
            try:
                plot_results(populated, plots / window / "ofc_conditioning_decoding", window)
            except Exception as exc:
                failures.append(f"{window} plotting: {exc}")
    config.update(run_status="failed" if failures else "complete", failures=failures)
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Results and QC: {output}", flush=True)
    if failures:
        raise RuntimeError("Decoding finished with failures; inspect qc.csv:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
