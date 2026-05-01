# CENG463 Midterm Takehome

This repository contains solutions for a CENG463 machine learning midterm takehome. The work is organized by question (q1 through q5) and uses Python scripts to run analyses, train models, and generate outputs.

## Repository layout

- q1/ - California housing EDA, feature engineering, modeling, and diagnostics
- q2/ - Imbalanced classification, calibration, and threshold tuning
- q3/ - Dimensionality reduction and autoencoder experiments on Fashion-MNIST
- q4/ - Clustering analysis, tuning, and evaluation
- q5/ - CIFAR-10 classification, evaluation, interpretability, and adversarial analysis
- data/ - Local datasets (e.g., CIFAR-10 batches)
- outputs/ - Saved results and figures per question

## Setup

1) Create and activate a virtual environment (if you do not already have one).

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies (there is no requirements.txt in this repo).

```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch torchvision
```

## How to run

Run each question from the repository root. Each question has a main entry point.

### Q1

```bash
python q1/main_q1.py
```

### Q2

```bash
python q2/main_q2.py
```

### Q3

```bash
python q3/main_q3.py
```

### Q4

```bash
python q4/main_q4.py
```

### Q5

```bash
python q5/main_q5.py
```

## Outputs

Results are written to the outputs/ directory, grouped by question. Some scripts may also display plots.

## Data notes

- CIFAR-10 data is expected under data/cifar-10-batches-py.
- The CIFAR-10 batches and tarball are gitignored, so you need to place them locally.
- Download CIFAR-10 (Python version) from https://www.cs.toronto.edu/~kriz/cifar.html and extract the tarball so the batches land in data/cifar-10-batches-py.
- Other datasets are downloaded or loaded in the corresponding question scripts.

## Troubleshooting

- If a script fails due to missing data, check the data/ folder paths in that question.
- If you see CUDA errors for Q5 or Q3, ensure your PyTorch installation matches your system or run on CPU.
