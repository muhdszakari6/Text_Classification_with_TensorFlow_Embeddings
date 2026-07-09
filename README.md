# Text Classification with TensorFlow Embeddings

TensorFlow-based text classification of GitHub pull requests, starting from raw
text and comparing three pretrained NNLM embedding spaces from TensorFlow Hub.

## Overview

Pull requests from the [numpy/numpy](https://github.com/numpy/numpy) GitHub
repository are collected via the GitHub API, cleaned and normalized, and used
to train a classifier that predicts a pull request's category — **Bug**,
**Enhancement**, **Documentation**, or **Testing** — from its title and body
text alone. Three pretrained embedding layers are trained end-to-end and
compared on accuracy, precision/recall per class, and training time.

## Files

| File | Description |
|---|---|
| `Text_Classification_with_TensorFlow_Embeddings.ipynb` | Full pipeline: data collection, preprocessing, model training, and evaluation. |
| `pull_requests.csv` | Raw dataset collected from the NumPy GitHub repository via the GitHub API. Columns: `id`, `label`, `title`, `body`. |
| `pull_requests_preprocessed.csv` | Cleaned dataset after text preprocessing. Columns: `id`, `label_encoded`, `model_input`. |

## Data collection

Closed pull requests are pulled from the NumPy repo using the GitHub API
(`ghapi`), keeping only PRs that:
- carry one of the labels `00 - Bug`, `01 - Enhancement`, `04 - Documentation`, or `05 - Testing`
- are not backports (title/body starting with "backport")
- were not opened by a bot account

Each PR's `body` is truncated to 10,000 characters before being written to
`pull_requests.csv`.

## Preprocessing

Title and body are combined and cleaned in `pull_requests_preprocessed.csv`:
1. Strip common ticket-style title prefixes (e.g. `ENH:`)
2. Remove Markdown code blocks
3. Expand contractions (`contractions`)
4. Strip HTML tags and URLs
5. Remove special characters and collapse whitespace
6. Lowercase the result

Labels are integer-encoded: `Bug=0`, `Enhancement=1`, `Documentation=2`, `Testing=3`.

## Modeling

Each model is a `tf_keras.Sequential` network built on a trainable TensorFlow
Hub embedding layer, followed by `Dense(16, relu)` → `Dropout(0.4)` →
`Dense(4, softmax)`, compiled with Adam and sparse categorical cross-entropy.

The three embedding spaces compared:

| Model | Embedding |
|---|---|
| NNLM 50D | `google/nnlm/en-dim50` |
| NNLM 128D | `google/nnlm/en-dim128` |
| NNLM 50D (normalized) | `google/nnlm/en-dim50-with-normalization` |

Class imbalance is handled by upsampling the minority classes (Enhancement,
Testing) to match the majority class (Bug), plus computed class weights
during training. Training uses an 80/20 stratified split, batch size 128, up
to 10 epochs, with early stopping on validation loss (patience 2, best
weights restored).

## Results

| Model | Test accuracy | Training time |
|---|---|---|
| NNLM 50D | 0.88 | ~67s |
| NNLM 128D | 0.90 | ~490s |
| NNLM 50D (normalized) | 0.90 | ~66s |

The normalized 50-dimensional embedding matches the 128-dimensional model's
accuracy at roughly 1/7th of the training time. Across all three models,
"Testing" is the easiest class to separate (F1 ≥ 0.95) and "Bug" the hardest
(F1 0.83–0.85), likely due to overlapping vocabulary between bug reports and
enhancement requests.

## Requirements

```
ghapi
pandas
matplotlib
tensorflow
tensorflow_hub
tensorflow_datasets
tf_keras
contractions
scikit-learn
```

## Running the notebook

1. Install the dependencies above.
2. Set a valid `GITHUB_TOKEN` environment variable (a GitHub personal access
   token) if you want to re-run data collection against the GitHub API — the
   token currently hardcoded in the notebook is a security risk and should be
   revoked and replaced with an environment-provided value before sharing or
   re-running this notebook.
3. Run all cells top to bottom. `pull_requests.csv` and
   `pull_requests_preprocessed.csv` are already included, so the notebook can
   also be run starting from the preprocessing/modeling cells without hitting
   the GitHub API.
# Text_Classification_with_TensorFlow_Embeddings
# Text_Classification_with_TensorFlow_Embeddings
