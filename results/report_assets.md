# Report Assets Index

## Section 3 — Data Analysis
- ![01 Class Distribution Full](figures/eda/01_class_distribution_full.png)
  Caption: The labeled subset is dominated by `none`, so accuracy alone would mostly measure majority-class performance.
- ![02 Class Distribution Defects Only](figures/eda/02_class_distribution_defects_only.png)
  Caption: Removing `none` exposes the long-tailed defect distribution, motivating macro-F1 and per-class metrics.
- ![03 Wafer Size Distribution](figures/eda/03_wafer_size_distribution.png)
  Caption: WM-811K has many wafer-map sizes; the most common labeled size is 25x27 with 18,781 samples, so resizing is necessary.
- ![04 Defect Density By Class](figures/eda/04_defect_density_by_class.png)
  Caption: `Near-full` and `Random` are almost separable by defect density, while `Scratch` and `none` overlap by defect count.
- ![05 Mean Image Per Class](figures/eda/05_mean_image_per_class.png)
  Caption: Class mean images reveal spatial priors such as center concentration, edge rings, and sparse local defects.
- ![06 Inter Class Similarity](figures/eda/06_inter_class_similarity.png)
  Caption: Mean-image similarity predicts later errors, especially among visually overlapping local-defect and `none`-like classes.
- ![07 Umap Embedding](figures/eda/07_umap_embedding.png)
  Caption: Class-balanced UMAP shows `Edge-Ring`, `Donut`, and `Center` separate more naturally than `Edge-Loc`, `Loc`, `Scratch`, and `none`.
- ![08 Label Noise In None](figures/eda/08_label_noise_in_none.png)
  Caption: The `none` density histogram shows a nonzero tail, giving evidence for possible label noise in the majority class.
- ![08B Suspicious None Examples](figures/eda/08b_suspicious_none_examples.png)
  Caption: High-density `none` examples provide concrete candidates for manual label-noise review.

## Section 5 — Experiments & Results
- ![Per Class F1 Comparison](figures/experiments/per_class_f1_comparison.png)
  Caption: The money chart compares per-class F1 across all six controlled experiments.
- ![Training Curves](exp_A_cnn_scratch/training_curves.png)
  Caption: `exp_A_cnn_scratch` training curves show convergence over 35 epochs and the validation macro-F1 trajectory.
- ![Training Curves](exp_B_transfer/training_curves.png)
  Caption: `exp_B_transfer` training curves show convergence over 45 epochs and the validation macro-F1 trajectory.
- ![Training Curves](exp_C_class_weight/training_curves.png)
  Caption: `exp_C_class_weight` training curves show convergence over 23 epochs and the validation macro-F1 trajectory.
- ![Training Curves](exp_D_focal_loss/training_curves.png)
  Caption: `exp_D_focal_loss` training curves show convergence over 44 epochs and the validation macro-F1 trajectory.
- ![Training Curves](exp_E_augmentation/training_curves.png)
  Caption: `exp_E_augmentation` training curves show convergence over 29 epochs and the validation macro-F1 trajectory.
- ![Training Curves](exp_F_svm/training_curves.png)
  Caption: `exp_F_svm` is a non-iterative SVM baseline; the placeholder figure marks that no epoch-wise deep-learning curve exists.

## Section 6 — Discussion
- ![Gradcam Per Class](figures/experiments/gradcam_per_class.png)
  Caption: Grad-CAM examples from `exp_E_augmentation` show confident correct cases and high-confidence failure modes for every class.
- ![Confusion Pair Edge Loc Vs None](figures/experiments/confusion_pair_Edge_Loc_vs_none.png)
  Caption: Case study of `Edge-Loc` wafers misclassified as `none`, illustrating a dominant confusion mode.
- ![Confusion Pair Loc Vs None](figures/experiments/confusion_pair_Loc_vs_none.png)
  Caption: Case study of `Loc` wafers misclassified as `none`, illustrating a dominant confusion mode.
- ![Likely Mislabeled None](figures/experiments/likely_mislabeled_none.png)
  Caption: The best CNN finds 5 `none`-labeled test wafers predicted as a defect class with confidence above 0.95, making them label-noise candidates.
