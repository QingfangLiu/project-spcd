#!/usr/bin/env python3
"""Run conditioning odor-identity RSA for EARLY and EXT5s cue estimates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

MRI_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MRI_SCRIPTS))
from rsa_imagewise_odor_identity_no_image_first_image import run_roi_rsa as roi
from rsa_imagewise_odor_identity_no_image_first_image.aggregate_roi_rsa_spearman import (
    _one_sample_summary,
)
from rsa_imagewise_odor_identity_no_image_first_image.run_ofc_cluster_map_k2 import CLUSTERS
from rsa_imagewise_odor_identity_no_image_first_image.run_roi_rsa_spearman import (
    SPEARMAN_COL,
    RANK_BISERIAL_COL,
    compute_rank_rsa_statistics,
)

WINDOWS = {
    "cueEARLY": "odorpred_cueEARLY_trialwise_2mm",
    "cueEXT5s": "odorpred_cueEXT5s_trialwise_2mm",
}
ANALYSES = {
    "primary_runs2-3": (2, 3),
    "sensitivity_runs1-3": (1, 2, 3),
}
PAIR_SCOPES = ("cross_run", "cross_run_cross_set")
DEFAULT_SKIP_SUBS = ("sub-19", "sub-31", "sub-33")
CUES = {2, 4, 6, 8, 10, 12}
QC_COLUMNS = ["sub", "window", "analysis", "roi", "status", "reason"]


def integer_columns(frame: pd.DataFrame, names, source: Path) -> pd.DataFrame:
    frame = frame.copy()
    for name in names:
        values = pd.to_numeric(frame[name], errors="raise")
        if values.isna().any() or not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError(f"Invalid integer column {name!r}: {source}")
        frame[name] = values.astype(int)
    return frame


def require_columns(frame: pd.DataFrame, names, source: Path) -> None:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}: {source}")


def load_behavior(behavior_dir: Path, sub: str) -> pd.DataFrame:
    path = behavior_dir / sub / "odor_pred_dat.csv"
    frame = pd.read_csv(path).rename(columns={"Run": "run"})
    require_columns(frame, ["run", "trial", "cue_idx", "odor_num", "resp", "acc", "rt"], path)
    frame = integer_columns(frame, ["run", "trial", "cue_idx", "odor_num"], path)
    if frame.duplicated(["run", "trial"]).any():
        raise ValueError(f"Duplicate behavioral run/trial keys: {path}")
    for name in ("resp", "acc", "rt"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not frame["acc"].isin([0, 1]).all():
        raise ValueError(f"Accuracy must be 0 or 1: {path}")
    responded = frame["resp"].notna()
    if not frame.loc[responded, "resp"].isin([0, 1, 2]).all():
        raise ValueError(f"Unexpected response positions: {path}")
    return frame.rename(columns={
        name: f"behavior_{name}" for name in ("cue_idx", "odor_num", "resp", "acc", "rt")
    })


def odor_partition_relation(meta: pd.DataFrame) -> str:
    """Accept a consistent global relabeling, never a change in odor grouping."""
    if not meta["cue_idx"].eq(meta["behavior_cue_idx"]).all():
        raise ValueError("Beta index and behavior disagree on cue identity.")
    if not set(meta["cue_idx"]).issubset(CUES) or not meta["odor_num"].isin([0, 1, 2]).all():
        raise ValueError("Unexpected conditioning cue or odor code.")
    mapping = meta.groupby("cue_idx")["odor_num"].agg(lambda x: set(x))
    if any(len(values) != 1 for values in mapping):
        raise ValueError("A conditioning cue changes odor labels across trials/runs.")
    mapping = {cue: next(iter(values)) for cue, values in mapping.items()}
    for cue, odor in mapping.items():
        if (cue in {6, 12}) != (odor == 0):
            raise ValueError("Clean-air and food-odor cue labels are inconsistent.")
    for first, second in ((2, 8), (4, 10)):
        if first in mapping and second in mapping and mapping[first] != mapping[second]:
            raise ValueError("Paired cue sets disagree on their associated odor.")
    for first, second in ((2, 4), (8, 10)):
        if first in mapping and second in mapping and mapping[first] == mapping[second]:
            raise ValueError("The two food cues within a set have the same odor label.")
    expected = meta["behavior_odor_num"]
    if meta["odor_num"].eq(expected).all():
        return "identical"
    if meta["odor_num"].eq(expected.replace({1: 2, 2: 1})).all():
        return "global_food_label_swap"
    raise ValueError("Beta/behavior odor labels differ beyond a global 1/2 swap.")


def load_window_samples(base_dir: Path, behavior: pd.DataFrame, sub: str,
                        window: str, runs: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    folder = base_dir / "GLM" / WINDOWS[window] / sub
    frames, missing, known_excluded = [], [], []
    for run in runs:
        # Exclude the entire interrupted scan, including its nominal late betas.
        if sub == "sub-35" and run == 3:
            known_excluded.append(run)
            continue
        path = folder / f"{sub}_odorpred_run-{run}_beta_index.csv"
        if not path.is_file():
            missing.append(run)
            continue
        frame = pd.read_csv(path)
        require_columns(frame, ["beta_file", "run", "trial", "trial_type", "cue_idx", "odor_num"], path)
        frame = frame.loc[frame["trial_type"].eq("cue")].copy()
        if frame.empty:
            raise ValueError(f"No cue rows in beta index: {path}")
        frame = integer_columns(frame, ["run", "trial", "cue_idx", "odor_num"], path)
        if not frame["run"].eq(run).all() or frame.duplicated(["run", "trial"]).any():
            raise ValueError(f"Inconsistent/duplicate run/trial keys: {path}")
        if frame["beta_file"].isna().any() or frame["beta_file"].duplicated().any():
            raise ValueError(f"Missing/duplicate beta filenames: {path}")
        frame["beta_path"] = [str(folder / str(name)) for name in frame["beta_file"]]
        frames.append(frame[["run", "trial", "cue_idx", "odor_num", "beta_path"]])
    qc = {
        "requested_runs": ",".join(map(str, runs)),
        "missing_index_runs": ",".join(map(str, missing)),
        "known_excluded_runs": ",".join(map(str, known_excluded)),
    }
    if not frames:
        return pd.DataFrame(), {**qc, "n_index_trials": 0}
    meta = pd.concat(frames, ignore_index=True)
    meta = meta.merge(behavior, on=["run", "trial"], how="left", validate="one_to_one", indicator=True)
    if not meta["_merge"].eq("both").all():
        raise ValueError(f"Some beta trials have no behavioral match: {sub} {window}")
    qc["odor_label_relation"] = odor_partition_relation(meta)
    food = meta["odor_num"].isin([1, 2])
    responded = meta["behavior_resp"].notna() & np.isfinite(meta["behavior_rt"]) & meta["behavior_rt"].gt(0)
    correct = meta["behavior_acc"].eq(1)
    qc.update(n_index_trials=len(meta), n_clean_air=int((~food).sum()),
              n_food_trials=int(food.sum()),
              n_food_missed=int((food & meta["behavior_resp"].isna()).sum()),
              n_food_invalid_rt=int((food & meta["behavior_resp"].notna() & ~responded).sum()),
              n_food_incorrect_responded=int((food & responded & ~correct).sum()))
    selected = meta.loc[food & responded & correct].drop(columns="_merge").copy()
    selected["cue_set"] = np.where(selected["cue_idx"] <= 6, 1, 2)
    return selected, qc


def matched_samples(base_dir: Path, behavior_dir: Path, sub: str, windows,
                    runs: tuple[int, ...]) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    behavior = load_behavior(behavior_dir, sub)
    frames, quality = {}, {}
    for window in windows:
        frames[window], quality[window] = load_window_samples(base_dir, behavior, sub, window, runs)
    if any(frame.empty for frame in frames.values()):
        return {window: pd.DataFrame() for window in windows}, quality
    keys = [set(zip(frame["run"], frame["trial"])) for frame in frames.values()]
    common = set.intersection(*keys)
    for window, frame in frames.items():
        keep = [(run, trial) in common for run, trial in zip(frame["run"], frame["trial"])]
        quality[window]["n_unmatched_window_trials"] = int(len(frame) - sum(keep))
        frames[window] = frame.loc[keep].sort_values(["run", "trial"]).reset_index(drop=True)
    reference = next(iter(frames.values()))
    for window, frame in frames.items():
        if not frame[["run", "trial", "cue_idx", "odor_num"]].equals(
            reference[["run", "trial", "cue_idx", "odor_num"]]
        ):
            raise ValueError(f"EARLY/EXT5s disagree on matched cue/odor labels: {sub}")
        # Check both windows before computing either, preserving matched sampling.
        missing = [path for path in frame["beta_path"] if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Selected beta image missing: {missing[0]}")
    return frames, quality


def eligibility(frame: pd.DataFrame, runs: tuple[int, ...], primary: bool) -> str:
    present = set(frame["run"]) if not frame.empty else set()
    if len(present) < 2:
        return "fewer_than_two_usable_runs"
    if primary and present != set(runs):
        return "missing_primary_run"
    counts = frame.groupby(["run", "odor_num"]).size().unstack(fill_value=0).reindex(columns=[1, 2], fill_value=0)
    if (counts == 0).any().any():
        return "a_run_has_no_correct_trials_for_one_odor"
    return ""


def analyze_patterns(samples: pd.DataFrame, raw_patterns: np.ndarray):
    run_ids = samples["run"].to_numpy()
    raw_used, patterns, used_mask, voxel_qc = roi._prepare_analysis_patterns(
        raw_patterns, run_ids, center_within_run=True, drop_nonfinite_voxels=True,
    )
    if patterns.shape[1] < 2 or np.any(np.linalg.norm(patterns - patterns.mean(axis=1, keepdims=True), axis=1) == 0):
        raise ValueError("Insufficient voxels or a constant centered sample pattern.")
    neural = roi.compute_correlation_rdm(patterns)
    odors = samples["odor_num"].to_numpy()
    model = (odors[:, None] != odors[None, :]).astype(float)
    cross_run = roi._cross_run_pair_mask(run_ids)
    sets = samples["cue_set"].to_numpy()
    tri = np.tril_indices(len(samples), k=-1)
    cross_set = (sets[:, None] != sets[None, :])[tri]
    masks = {"cross_run": cross_run, "cross_run_cross_set": cross_run & cross_set}
    results = {
        scope: {**compute_rank_rsa_statistics(neural, model, pair_mask),
                "n_pairs": int(pair_mask.sum())}
        for scope, pair_mask in masks.items()
    }
    return results, (raw_used, patterns, used_mask, voxel_qc, neural, model, masks)


def save_diagnostics(folder: Path, samples: pd.DataFrame, raw_patterns: np.ndarray,
                     details, roi_mask: np.ndarray) -> None:
    raw_used, patterns, used_mask, voxel_qc, neural, model, masks = details
    folder.mkdir(parents=True, exist_ok=True)
    samples.to_csv(folder / "samples.csv", index=False)
    np.savez_compressed(
        folder / "patterns_and_rdms.npz", raw_patterns=raw_patterns,
        raw_used_patterns=raw_used, analysis_patterns=patterns,
        roi_voxel_indices=np.column_stack(np.where(roi_mask)), used_voxel_mask=used_mask,
        neural_rdm=neural, odor_identity_rdm=model, **masks,
    )
    tri = np.tril_indices(len(samples), k=-1)
    pair_table = pd.DataFrame({
        "sample_i": tri[0], "sample_j": tri[1],
        "neural_distance": neural[tri], "different_odor": model[tri], **masks,
    })
    pair_table.loc[pair_table["cross_run"]].to_csv(folder / "rdm_pairs.csv", index=False)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, frame in results.groupby("pair_scope", sort=False):
        for metric in (SPEARMAN_COL, RANK_BISERIAL_COL):
            row = _one_sample_summary(
                frame[metric].to_numpy(), analysis_set=str(frame["analysis"].iloc[0]),
                test_label="one_sample", metric=metric, contrast=f"{scope}_gt_zero",
            )
            row.update(window=frame["window"].iloc[0], roi=frame["roi"].iloc[0], pair_scope=scope)
            rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def plot_results(results: pd.DataFrame, folder: Path, window: str, analysis: str, spec) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_conditioning_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(MRI_SCRIPTS.parent))
    from figure_style import configure_matplotlib, style_axis

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    wide = results.pivot(index="sub", columns="pair_scope", values=SPEARMAN_COL).reindex(columns=PAIR_SCOPES)
    offsets = np.linspace(-0.10, 0.10, len(wide))
    for i, (_, row) in enumerate(wide.iterrows()):
        ax.plot(np.arange(2) + offsets[i], row.to_numpy(), color="#b0b0b0", alpha=0.55, linewidth=0.7)
    for x, scope in enumerate(PAIR_SCOPES):
        values = wide[scope].dropna().to_numpy()
        ax.scatter(x + offsets, wide[scope], s=26, color=("#2878B5", "#D97721")[x], zorder=3)
        if len(values):
            mean = values.mean()
            ci = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values) if len(values) > 1 else 0
            ax.errorbar(x, mean, yerr=ci, fmt="D", mfc="white", mec="#29343D", ecolor="#29343D", capsize=5, zorder=5)
    ax.axhline(0, color="#7A858D", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1], ["Across runs", "Across runs\nand cue sets"])
    ax.set_ylabel("Odor-identity RSA (Spearman ρ)")
    ax.set_xlim(-0.45, 1.45)
    timing = "0–2 s" if window == "cueEARLY" else "0–5 s"
    selection = "Primary: runs 2–3" if analysis.startswith("primary") else "Sensitivity: runs 1–3"
    ax.set_title(f"{spec.plot_title}\n{window} ({timing}); {selection}", fontsize=13)
    fig.text(0.98, 0.015, f"n = {len(wide)}; diamonds: mean ± 95% CI", ha="right", fontsize=10)
    style_axis(ax)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{spec.key}.{extension}", dpi=200)
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_dir", type=Path, required=True, help="MRI data root on the data machine.")
    p.add_argument("--project_root", type=Path, default=MRI_SCRIPTS.parent)
    p.add_argument("--behavior_dir", type=Path, help="Default: <project_root>/beh_processed_subs.")
    p.add_argument("--mask_dir", type=Path, help="Default: prepared bilateral Kahnt K=2 masks in the repo.")
    p.add_argument("--repo_output_dir", type=Path, help="Default: <project_root>/rsa_roi/conditioning_odor_identity.")
    p.add_argument("--plot_dir", type=Path, help="Default: <project_root>/plots/conditioning_odor_identity.")
    p.add_argument("--diagnostics_dir", type=Path, help="Default: <base_dir>/rsa_roi/conditioning_odor_identity/diagnostics.")
    p.add_argument("--windows", nargs="+", choices=WINDOWS, default=list(WINDOWS))
    p.add_argument("--analyses", nargs="+", choices=ANALYSES, default=list(ANALYSES))
    p.add_argument("--clusters", nargs="+", choices=CLUSTERS, default=list(CLUSTERS))
    p.add_argument("--subjects", nargs="+", help="Default: discover subjects across selected GLM folders.")
    p.add_argument("--skip_subs", nargs="*", default=list(DEFAULT_SKIP_SUBS))
    p.add_argument("--save_diagnostics", action="store_true")
    p.add_argument("--skip_plots", action="store_true")
    p.add_argument("--strict", action="store_true", help="Stop at the first unexpected input/analysis failure.")
    p.add_argument("--dry_run", action="store_true", help="Print configuration without reading MRI data or writing outputs.")
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    project = args.project_root.expanduser().resolve()
    base = args.base_dir.expanduser().resolve()
    behavior_dir = (args.behavior_dir or project / "beh_processed_subs").expanduser().resolve()
    mask_dir = (args.mask_dir or project / "masks" / "OFC_clusterMap_K2_gm0p1").expanduser().resolve()
    output = (args.repo_output_dir or project / "rsa_roi" / "conditioning_odor_identity").expanduser().resolve()
    plots = (args.plot_dir or project / "plots" / "conditioning_odor_identity").expanduser().resolve()
    diagnostics = (args.diagnostics_dir or base / "rsa_roi" / "conditioning_odor_identity" / "diagnostics").expanduser().resolve()
    config = {
        "base_dir": str(base), "behavior_dir": str(behavior_dir), "mask_dir": str(mask_dir),
        "output_dir": str(output), "plot_dir": str(plots), "diagnostics_dir": str(diagnostics),
        "windows": {w: WINDOWS[w] for w in args.windows},
        "analyses": {a: ANALYSES[a] for a in args.analyses},
        "clusters": args.clusters, "subjects": args.subjects, "skip_subs": args.skip_subs,
        "trial_selection": "correct responded food-odor trials; matched across selected windows",
        "pair_scopes": PAIR_SCOPES, "center_within_run": True,
        "save_diagnostics": args.save_diagnostics, "skip_plots": args.skip_plots,
        "known_scan_exclusion": "sub-35 run 3",
    }
    print(json.dumps(config, indent=2))
    if args.dry_run:
        return
    for window in args.windows:
        folder = base / "GLM" / WINDOWS[window]
        if not folder.is_dir():
            raise FileNotFoundError(f"Conditioning GLM folder not found: {folder}")
    gm_path = base / "masks" / "gm_0.1.nii"
    if not gm_path.is_file():
        raise FileNotFoundError(f"Gray-matter mask not found: {gm_path}")
    masks = {
        name: roi._load_mask(mask_dir / CLUSTERS[name].mask_filename, gm_path)
        for name in args.clusters
    }
    subjects = sorted(set(args.subjects)) if args.subjects else sorted({
        path.name for window in args.windows
        for path in (base / "GLM" / WINDOWS[window]).iterdir()
        if path.is_dir() and path.name.startswith("sub-")
    })
    if not subjects:
        raise RuntimeError("No conditioning subjects found.")
    output.mkdir(parents=True, exist_ok=True)
    config["discovered_subjects"] = subjects
    (output / "run_manifest.json").write_text(json.dumps(config, indent=2) + "\n")
    bundles, errors = {}, []
    for analysis in args.analyses:
        runs = ANALYSES[analysis]
        for window in args.windows:
            for cluster in args.clusters:
                bundles[window, analysis, cluster] = {"rows": [], "qc": []}
        for sub in subjects:
            base_qc = {"sub": sub}
            if sub in args.skip_subs:
                frames, quality, status, reason = {}, {}, "excluded", "prespecified_subject_exclusion"
            else:
                try:
                    frames, quality = matched_samples(base, behavior_dir, sub, args.windows, runs)
                    reason = eligibility(next(iter(frames.values())), runs, analysis.startswith("primary"))
                    status = "ineligible" if reason else "ok"
                except Exception as exc:
                    if args.strict:
                        raise
                    frames, quality, status, reason = {}, {}, "failed", str(exc)
                    errors.append(f"{sub} {analysis}: {exc}")
                    print(f"[FAIL] {errors[-1]}")
            for window in args.windows:
                for cluster in args.clusters:
                    spec = CLUSTERS[cluster]
                    bundle = bundles[window, analysis, cluster]
                    qc = {**base_qc, "window": window, "analysis": analysis, "roi": spec.roi_name,
                          "status": status, "reason": reason, **quality.get(window, {})}
                    if status != "ok":
                        bundle["qc"].append(qc)
                        continue
                    samples = frames[window].copy()
                    qc.update(n_samples=len(samples), runs=",".join(map(str, sorted(samples["run"].unique()))),
                              n_runs=samples["run"].nunique(),
                              incomplete_requested_runs=set(samples["run"]) != set(runs),
                              trials_by_run_odor=json.dumps({f"run{r}_odor{o}": int(n) for (r, o), n in samples.groupby(["run", "odor_num"]).size().items()}))
                    try:
                        roi_img, roi_mask, *_ = masks[cluster]
                        raw = roi._load_roi_patterns(samples.to_dict("records"), roi_img, roi_mask)
                        values, details = analyze_patterns(samples, raw)
                        voxel_qc = details[3]
                        qc.update(**voxel_qc, voxel_loss_fraction=voxel_qc["n_voxels_dropped_nonfinite"] / voxel_qc["n_voxels_raw"])
                        qc["high_voxel_loss_20pct"] = qc["voxel_loss_fraction"] >= 0.20
                        if args.save_diagnostics:
                            save_diagnostics(diagnostics / window / analysis / spec.key / sub, samples, raw, details, roi_mask)
                        for scope, metrics in values.items():
                            bundle["rows"].append({**qc, "session": "odorpred", "pair_scope": scope,
                                                   "roi_mask": str(mask_dir / spec.mask_filename),
                                                   "odor_label_relation": quality[window]["odor_label_relation"], **metrics})
                        print(f"[OK] {sub} {window} {analysis} {cluster}: "
                              f"rho={values['cross_run'][SPEARMAN_COL]:.4f}; n={len(samples)}")
                    except Exception as exc:
                        if args.strict:
                            raise
                        qc.update(status="failed", reason=str(exc))
                        errors.append(f"{sub} {window} {analysis} {cluster}: {exc}")
                        print(f"[FAIL] {errors[-1]}")
                    bundle["qc"].append(qc)
    for bundle in bundles.values():
        bundle["results"] = pd.DataFrame(bundle["rows"])
        bundle["summary"] = summarize(bundle["results"]) if bundle["rows"] else pd.DataFrame()
    # Two prespecified lateral-OFC tests: one per window, primary runs, cross-run rho.
    family = []
    for (window, analysis, cluster), bundle in bundles.items():
        summary = bundle["summary"]
        if summary.empty:
            continue
        summary["p_primary_lateral_holm"] = np.nan
        if analysis == "primary_runs2-3" and cluster == "cluster2":
            index = summary.index[(summary["pair_scope"] == "cross_run") & (summary["metric"] == SPEARMAN_COL)][0]
            p_value = summary.loc[index, "p_greater_than_zero"]
            if np.isfinite(p_value):
                family.append((summary, index, p_value))
    # Keep the planned two-test family even if one window has no valid group test.
    adjusted = holm_adjust([item[2] for item in family] + [1.0] * (len(WINDOWS) - len(family)))
    for (summary, index, _), p_value in zip(family, adjusted):
        summary.loc[index, "p_primary_lateral_holm"] = p_value
    for (window, analysis, cluster), bundle in bundles.items():
        spec = CLUSTERS[cluster]
        folder = output / window / analysis / spec.key
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bundle["qc"], columns=list(dict.fromkeys(QC_COLUMNS + [key for row in bundle["qc"] for key in row]))).to_csv(folder / "qc.csv", index=False)
        bundle["results"].to_csv(folder / "subject_results.csv", index=False)
        bundle["summary"].to_csv(folder / "group_summary.csv", index=False)
        if not bundle["rows"]:
            errors.append(f"No valid results: {window} {analysis} {cluster}; inspect {folder / 'qc.csv'}")
        elif not args.skip_plots:
            plot_results(bundle["results"], plots / window / analysis, window, analysis, spec)
    print(f"Results and QC: {output}")
    if errors:
        raise RuntimeError("Analysis finished with failures; inspect qc.csv:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
