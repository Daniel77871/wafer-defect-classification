"""EDA figure generation for WM-811K."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.data.load import (
    LABELS,
    compute_wafer_stats,
    get_labeled_subset,
    load_wm811k,
)
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return resize(
        array,
        shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )


def _defect_mask(wafer_map: np.ndarray) -> np.ndarray:
    return (np.asarray(wafer_map) == 2).astype(np.float32)


def _save_class_distribution(df: pd.DataFrame, output_dir: Path) -> str:
    counts = df["label_str"].value_counts().reindex(LABELS, fill_value=0)
    dist = counts.rename_axis("label").reset_index(name="count")
    dist["percentage"] = dist["count"] / dist["count"].sum() * 100
    dist.to_csv(output_dir / "01_class_distribution_full.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.barplot(data=dist, x="label", y="count", ax=axes[0], color="#4C78A8")
    axes[0].set_title("WM-811K Labeled Class Distribution")
    axes[0].set_xlabel("Failure type")
    axes[0].set_ylabel("Number of wafers")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=dist, x="label", y="count", ax=axes[1], color="#F58518")
    axes[1].set_yscale("log")
    axes[1].set_title("Class Distribution with Log-Scaled Counts")
    axes[1].set_xlabel("Failure type")
    axes[1].set_ylabel("Number of wafers (log scale)")
    axes[1].tick_params(axis="x", rotation=45)
    _save_figure(fig, output_dir, "01_class_distribution_full")

    majority = dist.sort_values("count", ascending=False).iloc[0]
    minority = dist[dist["count"] > 0].sort_values("count").iloc[0]
    return (
        f"The labeled subset is dominated by `{majority['label']}`, which accounts "
        f"for {majority['percentage']:.1f}% of labeled wafers. The log-scale panel "
        f"makes the long tail visible; `{minority['label']}` has only "
        f"{int(minority['count'])} samples."
    )


def _save_defect_distribution(df: pd.DataFrame, output_dir: Path) -> str:
    defects = df[df["label_str"] != "none"]
    counts = defects["label_str"].value_counts().reindex(LABELS[:-1], fill_value=0)
    dist = counts.rename_axis("label").reset_index(name="count")
    dist["percentage_of_defects"] = dist["count"] / max(dist["count"].sum(), 1) * 100
    dist.to_csv(output_dir / "02_class_distribution_defects_only.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(data=dist, x="label", y="count", ax=ax, color="#54A24B")
    ax.set_title("Defect Class Distribution Excluding `none`")
    ax.set_xlabel("Failure type")
    ax.set_ylabel("Number of wafers")
    ax.tick_params(axis="x", rotation=45)
    _save_figure(fig, output_dir, "02_class_distribution_defects_only")

    largest = dist.sort_values("count", ascending=False).iloc[0]
    smallest = dist[dist["count"] > 0].sort_values("count").iloc[0]
    return (
        f"After excluding `none`, defect labels are still imbalanced: "
        f"`{largest['label']}` is the largest defect class and `{smallest['label']}` "
        f"is the smallest. This motivates macro-averaged metrics and per-class plots."
    )


def _save_size_distribution(eda_df: pd.DataFrame, output_dir: Path) -> str:
    size_counts = (
        eda_df.groupby(["height", "width"]).size().reset_index(name="count")
    )
    size_counts.to_csv(output_dir / "03_wafer_size_distribution.csv", index=False)
    top10 = size_counts.nlargest(10, "count")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    hist = ax.hist2d(
        eda_df["width"],
        eda_df["height"],
        bins=50,
        cmap="viridis",
        norm=LogNorm(),
    )
    fig.colorbar(hist[3], ax=ax, label="Number of wafers")
    top5 = top10.head(5)

    ax.scatter(
        top5["width"],
        top5["height"],
        color="red",
        s=35,
        label="Top 5 sizes",
    )
    ax.set_title("Wafer Map Size Distribution")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.legend()
    _save_figure(fig, output_dir, "03_wafer_size_distribution")

    common = top10.iloc[0]
    return (
        f"WM-811K contains variable-sized wafer maps; the most common labeled size is "
        f"{int(common['height'])}x{int(common['width'])} with {int(common['count'])} "
        f"samples. Resizing is therefore required before CNN training, and nearest-neighbor "
        f"interpolation preserves the discrete die values."
    )


def _save_density_by_class(eda_df: pd.DataFrame, output_dir: Path) -> str:
    density_df = eda_df[["label_str", "defect_density"]].copy()
    density_df.to_csv(output_dir / "04_defect_density_by_class.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    sns.boxplot(
        data=density_df,
        x="label_str",
        y="defect_density",
        order=LABELS,
        ax=ax,
        color="#72B7B2",
        whis=[5, 95],
        showfliers=True,
        fliersize=1.5,
        flierprops={
            "marker": ".",
            "markerfacecolor": "#2F4B7C",
            "markeredgecolor": "#2F4B7C",
            "alpha": 0.25,
        },
    )
    ax.set_ylim(0, 1.0)
    ax.set_title("Defect Density by Class")
    ax.set_xlabel("Failure type")
    ax.set_ylabel("Defective dies / total dies")
    ax.tick_params(axis="x", rotation=45)
    _save_figure(fig, output_dir, "04_defect_density_by_class")

    medians = density_df.groupby("label_str")["defect_density"].median()
    near_full_median = medians["Near-full"]
    random_median = medians["Random"]
    scratch_median = medians["Scratch"]
    none_median = medians["none"]
    return (
        f"Removing clipping exposes the real density structure: `Near-full` "
        f"(median {near_full_median:.2f}) and `Random` "
        f"(median {random_median:.2f}) are almost separable by the scalar defect "
        f"density alone, suggesting that a simple threshold baseline can solve these "
        f"classes. In contrast, `Scratch` (median {scratch_median:.2f}) and `none` "
        f"(median {none_median:.2f}) are not separable by defect count, which predicts "
        f"heavy CNN confusion and matches the high `Scratch`/`none` mean-image "
        f"similarity in Figure 6."
    )


def _compute_mean_images(df: pd.DataFrame, output_dir: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, float | int | str]] = []
    mean_images: dict[str, np.ndarray] = {}

    for label in tqdm(LABELS, desc="Mean images"):
        class_maps = df.loc[df["label_str"] == label, "waferMap"]
        accumulator = np.zeros((64, 64), dtype=np.float64)

        for wafer_map in class_maps:
            mask = _resize_nearest(_defect_mask(wafer_map), (64, 64))
            accumulator += mask

        if len(class_maps) > 0:
            accumulator /= len(class_maps)
        mean_images[label] = accumulator.astype(np.float32)

        for row_idx in range(64):
            for col_idx in range(64):
                rows.append(
                    {
                        "label": label,
                        "row": row_idx,
                        "col": col_idx,
                        "mean_defect_probability": float(
                            accumulator[row_idx, col_idx]
                        ),
                    }
                )

    pd.DataFrame(rows).to_csv(
        output_dir / "05_mean_image_per_class.csv",
        index=False,
    )
    return mean_images


def _save_mean_images(
    df: pd.DataFrame, output_dir: Path
) -> tuple[dict[str, np.ndarray], str]:
    mean_images = _compute_mean_images(df, output_dir)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    for ax, label in zip(axes.ravel(), LABELS):
        image = mean_images[label]
        im = ax.imshow(image, cmap="viridis", vmin=0, vmax=max(0.01, image.max()))
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Mean defect mask")
    fig.suptitle("Mean Defect Mask per Class After 64x64 Resizing")
    _save_figure(fig, output_dir, "05_mean_image_per_class")

    return (
        mean_images,
        "Mean images show the spatial prior each label encodes: edge classes concentrate "
        "near the wafer boundary, while center-like classes concentrate near the middle. "
        "These templates are useful for explaining later confusion patterns."
    )


def _save_inter_class_similarity(
    mean_images: dict[str, np.ndarray], output_dir: Path
) -> str:
    matrix = np.stack([mean_images[label].ravel() for label in LABELS])
    similarity = cosine_similarity(matrix)
    sim_df = pd.DataFrame(similarity, index=LABELS, columns=LABELS)
    sim_df.to_csv(output_dir / "06_inter_class_similarity.csv")

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    sns.heatmap(
        sim_df,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "Cosine similarity"},
    )
    ax.set_title("Inter-Class Similarity Between Mean Defect Masks")
    ax.set_xlabel("Failure type")
    ax.set_ylabel("Failure type")
    _save_figure(fig, output_dir, "06_inter_class_similarity")

    loc_edge = sim_df.loc["Loc", "Edge-Loc"]
    donut_center = sim_df.loc["Donut", "Center"]
    return (
        f"Mean-image cosine similarity gives a visual reason for expected mistakes: "
        f"`Loc`/`Edge-Loc` similarity is {loc_edge:.2f}, and `Donut`/`Center` "
        f"similarity is {donut_center:.2f}. These pairs should be watched closely in "
        f"confusion matrices."
    )


def _class_balanced_sample(
    df: pd.DataFrame, max_per_class: int, seed: int
) -> pd.DataFrame:
    sampled_parts: list[pd.DataFrame] = []

    for label in LABELS:
        class_df = df[df["label_str"] == label]
        if class_df.empty:
            continue
        sample_n = min(max_per_class, len(class_df))
        sampled_parts.append(class_df.sample(n=sample_n, random_state=seed))

    return pd.concat(sampled_parts).sample(frac=1, random_state=seed)


def _save_umap(df: pd.DataFrame, output_dir: Path, seed: int) -> str:
    from umap import UMAP

    sampled = _class_balanced_sample(df, max_per_class=500, seed=seed)
    vectors = np.stack(
        [
            (_resize_nearest(np.asarray(wafer_map), (32, 32)) / 2.0)
            .astype(np.float32)
            .ravel()
            for wafer_map in tqdm(sampled["waferMap"], desc="UMAP resize")
        ]
    )
    embedding = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=seed,
    ).fit_transform(vectors)

    umap_df = pd.DataFrame(
        {
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
            "label_str": sampled["label_str"].to_numpy(),
        }
    )
    umap_df.to_csv(output_dir / "07_umap_embedding.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    sns.scatterplot(
        data=umap_df,
        x="umap_x",
        y="umap_y",
        hue="label_str",
        hue_order=LABELS,
        s=8,
        alpha=0.6,
        linewidth=0,
        ax=ax,
    )
    ax.set_title("UMAP Projection with Class-Balanced Sampling")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.legend(
        title="Failure type",
        markerscale=2,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    _save_figure(fig, output_dir, "07_umap_embedding")

    return (
        "With class-balanced sampling, three classes form clear natural clusters: "
        "`Edge-Ring`, `Donut`, and `Center`. `Edge-Loc`, `Loc`, `Scratch`, and `none` "
        "collapse into the same central mass, giving a pre-experiment difficulty "
        "ranking that later per-class F1 scores should reflect."
    )


def _save_none_noise_examples(eda_df: pd.DataFrame, output_dir: Path) -> str:
    none_df = eda_df[eda_df["label_str"] == "none"].copy()
    if none_df.empty:
        threshold = math.nan
        fraction = math.nan
        suspicious = none_df
        hist_counts = np.array([], dtype=int)
        bin_edges = np.array([], dtype=float)
    else:
        threshold = float(none_df["defect_density"].quantile(0.99))
        suspicious_mask = none_df["defect_density"] > threshold
        fraction = float(suspicious_mask.mean())
        suspicious = none_df.loc[suspicious_mask].sort_values(
            "defect_density", ascending=False
        )
        hist_counts, bin_edges = np.histogram(none_df["defect_density"], bins=50)

    histogram_df = pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": hist_counts,
            "p99_threshold": threshold,
            "fraction_above_p99": fraction,
        }
    )
    histogram_df.to_csv(output_dir / "08_label_noise_in_none.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if not none_df.empty:
        ax.hist(none_df["defect_density"], bins=50, color="#B279A2", edgecolor="white")
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"99th percentile = {threshold:.4f}",
        )
        ax.legend()
    ax.set_title("Defect Density Distribution for `none` Class")
    ax.set_xlabel("Defective dies / total dies")
    ax.set_ylabel("Number of wafers")
    _save_figure(fig, output_dir, "08_label_noise_in_none")

    example_rows = suspicious.head(16)
    example_rows[
        ["defect_density", "height", "width", "defective_dies", "total_dies"]
    ].to_csv(output_dir / "08b_suspicious_none_examples.csv")

    fig, axes = plt.subplots(4, 4, figsize=(8, 8), constrained_layout=True)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    for ax, (index, row) in zip(axes.ravel(), example_rows.iterrows()):
        ax.imshow(np.asarray(row["waferMap"]), cmap="viridis", interpolation="nearest")
        ax.set_title(f"idx={index}\nd={row['defect_density']:.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Suspicious High-Density Examples Labeled `none`")
    _save_figure(fig, output_dir, "08b_suspicious_none_examples")

    pct = fraction * 100 if not math.isnan(fraction) else math.nan
    return (
        f"The `none` class still contains wafers with nonzero defect density. Using its "
        f"99th percentile ({threshold:.4f}) as a simple review threshold flags {pct:.2f}% "
        f"of `none` samples as suspicious, which is useful evidence for a label-noise "
        f"discussion."
    )


def run_all_eda(
    raw_path: str | Path = "data/raw/LSWMD.pkl",
    output_dir: str | Path = "results/figures/eda",
    seed: int = 42,
) -> None:
    """Generate all Phase 1 EDA artifacts."""
    set_seed(seed)
    output_path = ensure_dir(output_dir)

    df = load_wm811k(str(raw_path))
    labeled = get_labeled_subset(df)
    stats = compute_wafer_stats(labeled)
    eda_df = pd.concat(
        [labeled[["waferMap", "label_str", "label_id"]], stats],
        axis=1,
    )

    takeaways = {
        "01_class_distribution_full": _save_class_distribution(eda_df, output_path),
        "02_class_distribution_defects_only": _save_defect_distribution(
            eda_df, output_path
        ),
        "03_wafer_size_distribution": _save_size_distribution(eda_df, output_path),
        "04_defect_density_by_class": _save_density_by_class(eda_df, output_path),
    }
    mean_images, mean_takeaway = _save_mean_images(eda_df, output_path)
    takeaways["05_mean_image_per_class"] = mean_takeaway
    takeaways["06_inter_class_similarity"] = _save_inter_class_similarity(
        mean_images, output_path
    )
    takeaways["07_umap_embedding"] = _save_umap(eda_df, output_path, seed)
    takeaways["08_label_noise_in_none"] = _save_none_noise_examples(eda_df, output_path)

    with (output_path / "eda_takeaways.md").open("w", encoding="utf-8") as file:
        file.write("# EDA Takeaways\n\n")
        for name, takeaway in takeaways.items():
            file.write(f"## {name}\n\n{takeaway}\n\n")
