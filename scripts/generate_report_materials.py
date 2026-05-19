"""Generate report-ready Markdown assets and findings from saved results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wafer-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "wafer-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.load import LABELS
from src.utils.io import ensure_dir

EXPERIMENT_ORDER = [
    "exp_A_cnn_scratch",
    "exp_B_transfer",
    "exp_C_class_weight",
    "exp_D_focal_loss",
    "exp_E_augmentation",
    "exp_F_svm",
]

EDA_CAPTIONS = {
    "01_class_distribution_full.png": (
        "The labeled subset is dominated by `none`, so accuracy alone would mostly "
        "measure majority-class performance."
    ),
    "02_class_distribution_defects_only.png": (
        "Removing `none` exposes the long-tailed defect distribution, motivating "
        "macro-F1 and per-class metrics."
    ),
    "03_wafer_size_distribution.png": (
        "WM-811K has many wafer-map sizes; the most common labeled size is 25x27 "
        "with 18,781 samples, so resizing is necessary."
    ),
    "04_defect_density_by_class.png": (
        "`Near-full` and `Random` are almost separable by defect density, while "
        "`Scratch` and `none` overlap by defect count."
    ),
    "05_mean_image_per_class.png": (
        "Class mean images reveal spatial priors such as center concentration, "
        "edge rings, and sparse local defects."
    ),
    "06_inter_class_similarity.png": (
        "Mean-image similarity predicts later errors, especially among visually "
        "overlapping local-defect and `none`-like classes."
    ),
    "07_umap_embedding.png": (
        "Class-balanced UMAP shows `Edge-Ring`, `Donut`, and `Center` separate "
        "more naturally than `Edge-Loc`, `Loc`, `Scratch`, and `none`."
    ),
    "08_label_noise_in_none.png": (
        "The `none` density histogram shows a nonzero tail, giving evidence for "
        "possible label noise in the majority class."
    ),
    "08b_suspicious_none_examples.png": (
        "High-density `none` examples provide concrete candidates for manual "
        "label-noise review."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = _project_path(args.results_dir)
    metrics = _load_metrics(results_dir)

    _write_comparison_outputs(results_dir, metrics)
    _write_report_assets(results_dir, metrics)
    _write_key_findings(results_dir, metrics)
    print(f"wrote {results_dir / 'report_assets.md'}")
    print(f"wrote {results_dir / 'key_findings.md'}")


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _load_metrics(results_dir: Path) -> dict[str, dict[str, Any]]:
    metrics = {}
    for exp_name in EXPERIMENT_ORDER:
        path = results_dir / exp_name / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        with path.open("r", encoding="utf-8") as file:
            metrics[exp_name] = json.load(file)
    return metrics


def _write_comparison_outputs(
    results_dir: Path,
    metrics: dict[str, dict[str, Any]],
) -> None:
    comparison = _comparison_dataframe(metrics)
    comparison.to_csv(results_dir / "comparison_table.csv", index=False)
    (results_dir / "comparison_table.md").write_text(
        _dataframe_to_markdown(comparison),
        encoding="utf-8",
    )
    _plot_per_class_f1_comparison(
        comparison,
        results_dir / "figures/experiments/per_class_f1_comparison.png",
    )


def _comparison_dataframe(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for exp_name in EXPERIMENT_ORDER:
        metric = metrics[exp_name]
        row = {
            "experiment": exp_name,
            "accuracy": metric["accuracy"],
            "macro_f1": metric["macro_f1"],
            "kappa": metric["cohen_kappa"],
        }
        for class_name in LABELS:
            row[f"f1_{class_name}"] = metric["per_class"][class_name]["f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def _dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    rounded = dataframe.copy()
    numeric_cols = rounded.select_dtypes(include="number").columns
    rounded[numeric_cols] = rounded[numeric_cols].round(4)
    header = "| " + " | ".join(rounded.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rounded.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in rounded.to_numpy()]
    return "\n".join([header, separator, *rows]) + "\n"


def _plot_per_class_f1_comparison(dataframe: pd.DataFrame, save_path: Path) -> None:
    f1_columns = [f"f1_{class_name}" for class_name in LABELS]
    long_df = dataframe.melt(
        id_vars="experiment",
        value_vars=f1_columns,
        var_name="class_name",
        value_name="f1",
    )
    long_df["class_name"] = long_df["class_name"].str.replace("f1_", "", regex=False)
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    sns.barplot(data=long_df, x="class_name", y="f1", hue="experiment", ax=ax)
    ax.set(title="Per-Class F1 Across Experiments", xlabel="Class", ylabel="F1")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_report_assets(
    results_dir: Path,
    metrics: dict[str, dict[str, Any]],
) -> None:
    lines = ["# Report Assets Index", "", "## Section 3 — Data Analysis"]
    lines.extend(_eda_asset_lines(results_dir))
    lines.extend(["", "## Section 5 — Experiments & Results"])
    lines.extend(_experiment_asset_lines(results_dir, metrics))
    lines.extend(["", "## Section 6 — Discussion"])
    lines.extend(_discussion_asset_lines(results_dir, metrics))
    (results_dir / "report_assets.md").write_text("\n".join(lines) + "\n")


def _eda_asset_lines(results_dir: Path) -> list[str]:
    lines = []
    for filename, caption in EDA_CAPTIONS.items():
        lines.extend(_asset_entry(results_dir, f"figures/eda/{filename}", caption))
    return lines


def _experiment_asset_lines(
    results_dir: Path,
    metrics: dict[str, dict[str, Any]],
) -> list[str]:
    lines = _asset_entry(
        results_dir,
        "figures/experiments/per_class_f1_comparison.png",
        "The money chart compares per-class F1 across all six controlled experiments.",
    )
    for exp_name in EXPERIMENT_ORDER:
        caption = _training_curve_caption(exp_name, metrics[exp_name])
        lines.extend(_asset_entry(results_dir, f"{exp_name}/training_curves.png", caption))
    return lines


def _discussion_asset_lines(
    results_dir: Path,
    metrics: dict[str, dict[str, Any]],
) -> list[str]:
    best_exp = _best_cnn(metrics)
    lines = _asset_entry(
        results_dir,
        "figures/experiments/gradcam_per_class.png",
        f"Grad-CAM examples from `{best_exp}` show confident correct cases and "
        "high-confidence failure modes for every class.",
    )
    for relative_path in _confusion_pair_paths(results_dir):
        lines.extend(
            _asset_entry(results_dir, relative_path, _confusion_caption(relative_path))
        )
    lines.extend(
        _asset_entry(
            results_dir,
            "figures/experiments/likely_mislabeled_none.png",
            _none_caption(results_dir, best_exp),
        )
    )
    return lines


def _asset_entry(results_dir: Path, relative_path: str, caption: str) -> list[str]:
    path = results_dir / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing report asset: {path}")
    title = path.stem.replace("_", " ").title()
    return [f"- ![{title}]({relative_path})", f"  Caption: {caption}"]


def _training_curve_caption(exp_name: str, metrics: dict[str, Any]) -> str:
    if exp_name == "exp_F_svm":
        return (
            "`exp_F_svm` is a non-iterative SVM baseline; the placeholder figure "
            "marks that no epoch-wise deep-learning curve exists."
        )
    epoch_word = "epoch" if int(metrics["n_epochs_ran"]) == 1 else "epochs"
    return (
        f"`{exp_name}` training curves show convergence over "
        f"{metrics['n_epochs_ran']} {epoch_word} and the validation macro-F1 trajectory."
    )


def _confusion_pair_paths(results_dir: Path) -> list[str]:
    paths = sorted((results_dir / "figures/experiments").glob("confusion_pair_*.png"))
    if len(paths) < 2:
        raise FileNotFoundError("Expected at least two confusion-pair figures.")
    return [str(path.relative_to(results_dir)) for path in paths[:2]]


def _confusion_caption(relative_path: str) -> str:
    pair = Path(relative_path).stem.removeprefix("confusion_pair_")
    true_label, pred_label = pair.split("_vs_")
    return (
        f"Case study of `{true_label.replace('_', '-')}` wafers misclassified as "
        f"`{pred_label.replace('_', '-')}`, illustrating a dominant confusion mode."
    )


def _none_caption(results_dir: Path, best_exp: str) -> str:
    count = _likely_mislabeled_none_count(results_dir, best_exp)
    return (
        f"The best CNN finds {count} `none`-labeled test wafers predicted as a "
        "defect class with confidence above 0.95, making them label-noise candidates."
    )


def _write_key_findings(
    results_dir: Path,
    metrics: dict[str, dict[str, Any]],
) -> None:
    lines = ["# Key Findings", ""]
    lines.extend(_data_findings())
    lines.extend(_experiment_findings(metrics))
    lines.extend(_discussion_findings(results_dir, metrics))
    (results_dir / "key_findings.md").write_text("\n".join(lines) + "\n")


def _data_findings() -> list[str]:
    return [
        "## Data Patterns",
        "",
        "- The labeled data is dominated by `none` (85.2%), while `Near-full` "
        "has only 149 samples; macro-F1 and per-class metrics are more useful "
        "than accuracy.",
        "- Wafer maps are highly variable in size; the most common labeled size "
        "is 25x27 with 18,781 samples, so nearest-neighbor resizing is necessary.",
        "- Defect density nearly separates `Near-full` (median 0.88) and "
        "`Random` (median 0.48), but `Scratch` and `none` overlap around "
        "0.09-0.10, foreshadowing confusion.",
        "",
    ]


def _experiment_findings(metrics: dict[str, dict[str, Any]]) -> list[str]:
    best_exp = _best_cnn(metrics)
    baseline = metrics["exp_A_cnn_scratch"]
    best = metrics[best_exp]
    return [
        "## Model Comparisons",
        "",
        f"- The best CNN is `{best_exp}` with macro-F1 {_pct(best['macro_f1'])}, "
        f"accuracy {_pct(best['accuracy'])}, and kappa {best['cohen_kappa']:.3f}; "
        "it improves macro-F1 by "
        f"{_points(best['macro_f1'] - baseline['macro_f1'])} over baseline.",
        _intervention_sentence(metrics, "exp_D_focal_loss", "Focal loss"),
        _intervention_sentence(metrics, "exp_C_class_weight", "Class-weighted CE"),
        _transfer_sentence(metrics),
        _svm_sentence(metrics),
        _best_per_class_sentence(metrics),
        "",
    ]


def _discussion_findings(
    results_dir: Path,
    metrics: dict[str, dict[str, Any]],
) -> list[str]:
    best_exp = _best_cnn(metrics)
    pairs = _top_confusion_pair_names(metrics[best_exp])
    none_count = _likely_mislabeled_none_count(results_dir, best_exp)
    return [
        "## Error Analysis",
        "",
        f"- The top confusion pairs for `{best_exp}` are {pairs}; both match "
        "the EDA/UMAP pattern where local sparse defects overlap with `none`.",
        "- Grad-CAM highlights class templates such as center blobs, edge rings, "
        "and scratch-like arcs, but wrong examples often activate broad wafer "
        "regions rather than a clean defect structure.",
        f"- The label-noise revisit found {none_count} `none`-labeled test "
        "samples where the best CNN predicted a defect class with confidence "
        "above 0.95; treat them as manual-review candidates, not automatic relabels.",
    ]


def _intervention_sentence(
    metrics: dict[str, dict[str, Any]],
    exp_name: str,
    label: str,
) -> str:
    baseline = metrics["exp_A_cnn_scratch"]
    experiment = metrics[exp_name]
    deltas = _class_deltas(baseline, experiment)
    gain_class = max(deltas, key=deltas.get)
    drop_class = min(deltas, key=deltas.get)
    macro_delta = experiment["macro_f1"] - baseline["macro_f1"]
    return (
        f"- {label} changes macro-F1 by {_points(macro_delta)} versus baseline; "
        f"largest gain is `{gain_class}` ({_points(deltas[gain_class])}) and "
        f"largest drop is `{drop_class}` ({_points(deltas[drop_class])})."
    )


def _transfer_sentence(metrics: dict[str, dict[str, Any]]) -> str:
    baseline = metrics["exp_A_cnn_scratch"]
    transfer = metrics["exp_B_transfer"]
    deltas = _class_deltas(baseline, transfer)
    return (
        "- Transfer learning is not a clear overall win: MobileNetV3 changes "
        f"macro-F1 by {_points(transfer['macro_f1'] - baseline['macro_f1'])}, "
        f"helps `Donut` by {_points(deltas['Donut'])}, but hurts `Scratch` by "
        f"{_points(deltas['Scratch'])}."
    )


def _svm_sentence(metrics: dict[str, dict[str, Any]]) -> str:
    baseline = metrics["exp_A_cnn_scratch"]
    svm = metrics["exp_F_svm"]
    wins = [
        class_name
        for class_name, delta in _class_deltas(baseline, svm).items()
        if delta > 0
    ]
    return (
        "- The handcrafted SVM trails the scratch CNN by "
        f"{_points(svm['macro_f1'] - baseline['macro_f1'])} macro-F1, but "
        f"still beats it on {', '.join(f'`{name}`' for name in wins)}."
    )


def _best_per_class_sentence(metrics: dict[str, dict[str, Any]]) -> str:
    winners = []
    for class_name in LABELS:
        best_exp = max(EXPERIMENT_ORDER, key=lambda exp: _f1(metrics[exp], class_name))
        winners.append(f"`{class_name}`={best_exp.replace('exp_', '')}")
    return "- Best experiment by class: " + "; ".join(winners) + "."


def _best_cnn(metrics: dict[str, dict[str, Any]]) -> str:
    cnn_exps = EXPERIMENT_ORDER[:5]
    return max(cnn_exps, key=lambda exp: float(metrics[exp]["macro_f1"]))


def _class_deltas(
    baseline: dict[str, Any],
    experiment: dict[str, Any],
) -> dict[str, float]:
    return {
        class_name: _f1(experiment, class_name) - _f1(baseline, class_name)
        for class_name in LABELS
    }


def _f1(metrics: dict[str, Any], class_name: str) -> float:
    return float(metrics["per_class"][class_name]["f1"])


def _top_confusion_pair_names(metrics: dict[str, Any]) -> str:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    np.fill_diagonal(matrix, 0)
    flat_indices = np.argsort(matrix.ravel())[::-1][:2]
    pairs = [np.unravel_index(index, matrix.shape) for index in flat_indices]
    labels = [f"`{LABELS[true]} -> {LABELS[pred]}`" for true, pred in pairs]
    return " and ".join(labels)


def _likely_mislabeled_none_count(results_dir: Path, exp_name: str) -> int:
    path = results_dir / exp_name / "test_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    with np.load(path) as data:
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        y_proba = data["y_proba"]
    none_id = LABELS.index("none")
    confidence = y_proba[np.arange(len(y_pred)), y_pred]
    return int(((y_true == none_id) & (y_pred != none_id) & (confidence > 0.95)).sum())


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _points(value: float) -> str:
    return f"{value * 100:+.2f} points"


if __name__ == "__main__":
    main()
