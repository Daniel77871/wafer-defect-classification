# Key Findings

## Data Patterns

- The labeled data is dominated by `none` (85.2%), while `Near-full` has only 149 samples; macro-F1 and per-class metrics are more useful than accuracy.
- Wafer maps are highly variable in size; the most common labeled size is 25x27 with 18,781 samples, so nearest-neighbor resizing is necessary.
- Defect density nearly separates `Near-full` (median 0.88) and `Random` (median 0.48), but `Scratch` and `none` overlap around 0.09-0.10, foreshadowing confusion.

## Model Comparisons

- The best CNN is `exp_E_augmentation` with macro-F1 87.41%, accuracy 97.68%, and kappa 0.912; it improves macro-F1 by +2.39 points over baseline.
- Focal loss changes macro-F1 by +1.07 points versus baseline; largest gain is `Donut` (+5.24 points) and largest drop is `Near-full` (-1.84 points).
- Class-weighted CE changes macro-F1 by -5.83 points versus baseline; largest gain is `Donut` (+7.20 points) and largest drop is `Loc` (-19.07 points).
- Transfer learning is not a clear overall win: MobileNetV3 changes macro-F1 by -0.34 points, helps `Donut` by +5.43 points, but hurts `Scratch` by -18.40 points.
- The handcrafted SVM trails the scratch CNN by -5.11 points macro-F1, but still beats it on `Donut`, `Edge-Ring`, `Near-full`.
- Best experiment by class: `Center`=D_focal_loss; `Donut`=C_class_weight; `Edge-Loc`=E_augmentation; `Edge-Ring`=D_focal_loss; `Loc`=E_augmentation; `Near-full`=B_transfer; `Random`=B_transfer; `Scratch`=E_augmentation; `none`=E_augmentation.

## Error Analysis

- The top confusion pairs for `exp_E_augmentation` are `Edge-Loc -> none` and `Loc -> none`; both match the EDA/UMAP pattern where local sparse defects overlap with `none`.
- Grad-CAM highlights class templates such as center blobs, edge rings, and scratch-like arcs, but wrong examples often activate broad wafer regions rather than a clean defect structure.
- The label-noise revisit found 5 `none`-labeled test samples where the best CNN predicted a defect class with confidence above 0.95; treat them as manual-review candidates, not automatic relabels.
