# EDA Takeaways

## 01_class_distribution_full

The labeled subset is dominated by `none`, which accounts for 85.2% of labeled wafers. The log-scale panel makes the long tail visible; `Near-full` has only 149 samples.

## 02_class_distribution_defects_only

After excluding `none`, defect labels are still imbalanced: `Edge-Ring` is the largest defect class and `Near-full` is the smallest. This motivates macro-averaged metrics and per-class plots.

## 03_wafer_size_distribution

WM-811K contains variable-sized wafer maps; the most common labeled size is 25x27 with 18781 samples. Resizing is therefore required before CNN training, and nearest-neighbor interpolation preserves the discrete die values.

## 04_defect_density_by_class

Removing clipping exposes the real density structure: `Near-full` (median 0.88) and `Random` (median 0.48) are almost separable by the scalar defect density alone, suggesting that a simple threshold baseline can solve these classes. In contrast, `Scratch` (median 0.09) and `none` (median 0.10) are not separable by defect count, which predicts heavy CNN confusion and matches the high `Scratch`/`none` mean-image similarity in Figure 6.

## 05_mean_image_per_class

Mean images show the spatial prior each label encodes: edge classes concentrate near the wafer boundary, while center-like classes concentrate near the middle. These templates are useful for explaining later confusion patterns.

## 06_inter_class_similarity

Mean-image cosine similarity gives a visual reason for expected mistakes: `Loc`/`Edge-Loc` similarity is 0.94, and `Donut`/`Center` similarity is 0.74. These pairs should be watched closely in confusion matrices.

## 07_umap_embedding

With class-balanced sampling, three classes form clear natural clusters: `Edge-Ring`, `Donut`, and `Center`. `Edge-Loc`, `Loc`, `Scratch`, and `none` collapse into the same central mass, giving a pre-experiment difficulty ranking that later per-class F1 scores should reflect.

## 08_label_noise_in_none

The `none` class still contains wafers with nonzero defect density. Using its 99th percentile (0.2442) as a simple review threshold flags 0.96% of `none` samples as suspicious, which is useful evidence for a label-noise discussion.

