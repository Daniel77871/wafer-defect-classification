# Wafer Defect Classification

This project builds a reproducible PyTorch-based workflow for wafer-map defect classification on the WM-811K dataset. The emphasis is on problem framing, data analysis, comparable experiments, and interpretable artifacts such as per-class metrics, confusion matrices, mean wafer maps, Grad-CAM overlays, and error case studies rather than chasing the highest possible accuracy.

## How to Reproduce

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download WM-811K from Kaggle and place the raw pickle at:

   ```text
   data/raw/LSWMD.pkl
   ```

4. Run EDA:

   ```bash
   python scripts/run_eda.py
   ```

5. Run preprocessing once it is implemented:

   ```bash
   python scripts/run_preprocessing.py
   ```

6. Run one experiment:

   ```bash
   python scripts/run_experiment.py --config configs/exp_A_cnn_scratch.yaml
   ```

7. Run all configured experiments:

   ```bash
   bash scripts/run_all_experiments.sh
   ```

EDA outputs are written to `results/figures/eda/`. Experiment outputs will be written to `results/<exp_name>/`.

## Folder Structure

```text
wafer-defect-classification/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- splits/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_preprocessing_check.ipynb
|   `-- 03_results_analysis.ipynb
|-- src/
|   |-- data/
|   |   |-- load.py
|   |   |-- preprocess.py
|   |   `-- split.py
|   |-- features/
|   |   `-- handcrafted.py
|   |-- models/
|   |   |-- cnn_scratch.py
|   |   |-- transfer.py
|   |   `-- svm_baseline.py
|   |-- training/
|   |   |-- train.py
|   |   |-- losses.py
|   |   |-- augmentation.py
|   |   `-- sampler.py
|   |-- evaluation/
|   |   |-- metrics.py
|   |   |-- visualize.py
|   |   |-- gradcam.py
|   |   `-- eda.py
|   `-- utils/
|       |-- seed.py
|       |-- config.py
|       `-- io.py
|-- configs/
|-- scripts/
|-- results/
|-- requirements.txt
`-- README.md
```
