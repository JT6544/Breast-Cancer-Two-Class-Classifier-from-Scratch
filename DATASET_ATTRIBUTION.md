# Dataset attribution

This project includes the **Breast Cancer Wisconsin (Original)** dataset from
the UCI Machine Learning Repository.

- Creator: William H. Wolberg
- Dataset record: https://archive.ics.uci.edu/dataset/15/breast%2Bcancer%2Bwisconsin%2Boriginal
- DOI: https://doi.org/10.24432/C5HP4Z
- Citation: Wolberg, W. (1990). *Breast Cancer Wisconsin (Original)* [Dataset].
  UCI Machine Learning Repository.
- Dataset licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Licence text: https://creativecommons.org/licenses/by/4.0/

## Included file

`data/raw/breast-cancer-wisconsin.data` is the unmodified UCI raw data file
downloaded from:

https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data

Its SHA-256 checksum is recorded in `data/raw/SHA256SUMS`.

## Project transformation

The original dataset contains a sample code number, nine integer-valued
features, and a class label. This project reproduces the eight-feature scope of
the original coursework program by explicitly omitting `Bare Nuclei`, the only
feature containing missing values. The other eight features are retained in
their original order and scale.

Original classes are mapped explicitly:

- `2` becomes `0` (`benign`)
- `4` becomes `1` (`malignant`)

The sample code number is retained only to ensure repeated identifiers cannot
appear across training, validation, and test subsets. It is never supplied to
the classifier as a predictive feature.

Run `python scripts/prepare_data.py` to generate an optional conventional,
row-oriented CSV derivative. The raw source file is never overwritten.
