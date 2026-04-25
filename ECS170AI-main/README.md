# ECS170AI

## Evaluation Metrics

In this experiment, we use Accuracy, Precision, Recall, and F1 score to evaluate multiclass classification performance.  
For each class \(c\), we define:

- \(TP_c\): true positives of class \(c\)
- \(FP_c\): false positives of class \(c\)
- \(FN_c\): false negatives of class \(c\)
- \(N\): total number of samples

### 1) Accuracy

\[
\text{Accuracy}=\frac{\text{Number of correct predictions}}{N}
\]

Meaning: Accuracy measures overall correctness. It is intuitive, but it may hide poor performance on minority classes when class distribution is imbalanced.

### 2) Precision

For class \(c\):
\[
\text{Precision}_c=\frac{TP_c}{TP_c+FP_c}
\]

Meaning: Precision answers: "When the model predicts class \(c\), how often is it correct?"  
Higher precision means fewer false positives.

### 3) Recall

For class \(c\):
\[
\text{Recall}_c=\frac{TP_c}{TP_c+FN_c}
\]

Meaning: Recall answers: "Of all true samples in class \(c\), how many did the model find?"  
Higher recall means fewer false negatives.

### 4) F1 Score

For class \(c\):
\[
\text{F1}_c=\frac{2\cdot \text{Precision}_c\cdot \text{Recall}_c}{\text{Precision}_c+\text{Recall}_c}
\]

Meaning: F1 is the harmonic mean of Precision and Recall, so it is high only when both are high. It is useful when we want a balanced view between false positives and false negatives.

### Multiclass Averaging (used in this project)

Since this is a multiclass problem, Precision/Recall/F1 are aggregated across classes:

- **Macro average**:
\[
\text{Metric}_{macro}=\frac{1}{C}\sum_{c=1}^{C}\text{Metric}_c
\]
All classes are weighted equally; emphasizes minority-class performance.

- **Weighted average**:
\[
\text{Metric}_{weighted}=\sum_{c=1}^{C}\frac{n_c}{N}\text{Metric}_c
\]
Each class is weighted by its support \(n_c\) (number of true samples in class \(c\)); reflects class imbalance.

- **Micro average**:
\[
\text{Precision}_{micro}=\frac{\sum_c TP_c}{\sum_c (TP_c+FP_c)},\quad
\text{Recall}_{micro}=\frac{\sum_c TP_c}{\sum_c (TP_c+FN_c)}
\]
\[
\text{F1}_{micro}=\frac{2\cdot \text{Precision}_{micro}\cdot \text{Recall}_{micro}}
{\text{Precision}_{micro}+\text{Recall}_{micro}}
\]
Micro average pools all classes together before computing the metric, so it is dominated by overall sample-level performance.

In summary, we report Accuracy plus macro/weighted/micro Precision, Recall, and F1 to provide a complete and fair evaluation of multiclass model behavior.

## Evaluation Metrics Update

This project performs multiclass classification, so metrics are reported with multiclass averaging modes.

- Added `micro`-averaged metrics to stage 2 evaluation:
  - `Precision (micro)`
  - `Recall (micro)`
  - `F1 (micro)`
- Existing `macro` and `weighted` metrics are kept.
- Updated scripts:
  - `stage_2_code/run_stage2.py`
  - `stage_2_code/run_ablation.py`

## Run

From repo root:

- `python stage_2_code/run_stage2.py`
- `python stage_2_code/run_ablation.py`

All data and output paths are resolved from the script directory (`stage_2_code/`), so running from different current directories is supported.