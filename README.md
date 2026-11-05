# Penguin Perceptron

This project studies a single-layer perceptron for classifying Adelie, Gentoo,
and Chinstrap penguins from four morphological measurements.

The notebook is written in French. A French project overview is available in
[`README.fr.md`](./README.fr.md).

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running the project

Open [`penguin_perceptron_analysis.ipynb`](./penguin_perceptron_analysis.ipynb)
in Jupyter or VS Code, then select **Restart & Run All**.

The dependencies are listed in [`requirements.txt`](./requirements.txt).
The notebook uses seed `23` for reproducible data splits, noise generation,
and perceptron initialization.

## Data loading

Data loading is implemented by `src.data.load_data`. It accepts either a local
file path or a URL, so importing the package does not download data
automatically. This also makes offline imports possible.

## Implementation notes

- The perceptron has no explicit bias term.
- Training and validation use normalized features.
- Decision thresholds are selected on the validation set and then applied to
  the test set.
- ROC and precision-recall metrics are implemented for educational purposes.
- AUC and AP use trapezoidal integration. The code uses `numpy.trapezoid` when
  available and falls back to `numpy.trapz` for older NumPy versions.

## Project structure

```text
penguin_perceptron_analysis.ipynb
src/
  data.py
  metrics.py
  perceptron.py
  plots.py
  validation.py
requirements.txt
README.fr.md
```
