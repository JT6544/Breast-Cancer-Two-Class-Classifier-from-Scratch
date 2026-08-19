# Improvement Record: Breast-Cancer Two-Class Classifier — Gate 3

## Archive represented

This document describes the rebuilt implementation contained in:

```text
Breast-Cancer-Two-Class-Classifier-from-Scratch-Gate3.zip
```

The accompanying `Linear Two-Class Classifier - Breast Cancer Data Set.py` is the untouched original source used as the baseline. Its SHA-256 checksum is:

```text
1744a742d6245c39eafdbe40fcb21588cf298c34dc6907282916031f1d0c0c9c  Linear Two-Class Classifier - Breast Cancer Data Set.py
```

Gate 3 contains the corrected modelling workflow, tests, CI configuration, attribution, safety notice, reference metrics, and figures from Gate 2, plus the publication README. The classifier and verified results were not changed during the documentation gate.

## Executive summary

The original script demonstrated binary logistic classification using eight cytology features and Autograd. It trained and evaluated on the same observations, chose the “best” iteration using training misclassification count, used no feature scaling, and loaded a custom transposed CSV without formal provenance.

The rebuild preserves the eight-feature educational scope while replacing the evaluation design with grouped training, validation, and test subsets. Scaling is fitted only on training observations, weights are selected by validation loss, and the held-out test subset is used once. The official raw UCI dataset is bundled unchanged with checksum and attribution information.

Because the subject is medical, the rebuild also adds an explicit safety document and labels all outputs as educational rather than clinically validated.

## Original implementation limitations

- Training metrics and final metrics were calculated on the same data.
- The lowest training misclassification count selected the saved iteration.
- No independent validation or test observations existed.
- Repeated sample identifiers were unavailable to the split logic and could not be grouped.
- Features on different scales were used directly.
- `log(1 + exp(...))` and the sigmoid were evaluated without numerical stability guards.
- Autograd was required for a gradient that has a compact analytical form.
- The custom CSV was column-oriented and lacked source, checksum, and transformation documentation.
- Model outputs for example “patients” were presented without a strong non-clinical disclaimer.
- Model parameters, scaler values, metrics, and training history were not saved.
- The script executed immediately on import and displayed many interactive plots.
- There were no automated tests or continuous-integration checks.

## Improvement summary

| Area | Improvement | Why it was made | Impact |
|---|---|---|---|
| Dataset | Replaced the undocumented derivative CSV with the unchanged official UCI raw file | Provenance and transformations must be traceable | The repository records source, licence, DOI, checksum, and feature selection |
| Feature scope | Explicitly omitted `Bare Nuclei` while retaining the original eight predictors | That feature is the only one with missing values and was absent from the earlier CSV | The educational scope is preserved without silently imputing data |
| Labels | Mapped UCI class `2` to benign `0` and class `4` to malignant `1` | Positive-class semantics must be unambiguous | Confusion matrices and medical metrics use malignant disease consistently |
| Grouping | Retained sample code numbers only as group identifiers | Some identifiers repeat in the source data | One identifier cannot cross training, validation, and test subsets |
| Evaluation | Added deterministic 70/15/15 grouped stratified subsets | Training-set performance is optimistically biased | The final metrics come from 104 untouched test observations |
| Scaling | Fitted means and standard deviations using training data only | Global scaling leaks held-out information | Validation, test, and synthetic vectors use only training-derived preprocessing |
| Loss | Used stable logistic loss through `np.logaddexp` | Direct exponentiation can overflow | Extreme scores remain finite |
| Gradient | Replaced Autograd with the analytical gradient | The derivative is simple and should be inspectable | Autograd is removed and gradient correctness is unit-tested |
| Selection | Saved weights at the lowest validation loss with patience | Training misclassification is discrete and not independent evidence | Model selection no longer uses training error or test data |
| Metrics | Added precision, recall, specificity, F1, and balanced accuracy | Overall accuracy can hide class-specific errors | Malignant false negatives and benign false positives are visible |
| Outputs | Saved JSON metrics, CSV history, `.npz` model state, and figures | Console-only results are difficult to reproduce | Results can be audited and reused programmatically |
| Safety | Added `SAFETY.md` and explicit educational disclaimers | A coursework model is not a medical device | The repository reduces the risk of clinical misinterpretation |
| Quality | Added ten pipeline/model checks across two test modules and GitHub Actions | Numerical and data invariants need regression protection | Dataset structure, splitting, scaling, loss, gradient, and CLI behaviour are checked |

## Dataset and preprocessing improvements

### Traceable source data

The improved archive includes the unmodified UCI **Breast Cancer Wisconsin (Original)** file. It contains 699 observations with original class codes:

$$
2\equiv\text{benign},
\qquad
4\equiv\text{malignant}.
$$

The project mapping is

$$
y=
\begin{cases}
0,&\text{benign},\\
1,&\text{malignant}.
\end{cases}
$$

The mapping, feature removal, and row-oriented preparation route are documented in `DATASET_ATTRIBUTION.md` and `scripts/prepare_data.py`.

### Preserving the eight-feature coursework scope

The official source contains nine predictive features. `Bare Nuclei` is omitted explicitly because it contains `?` values and was absent from the original eight-feature input matrix. The rebuild does not hide this choice or silently discard whole observations.

The sample code number is not used as a model feature. It is preserved only to prevent repeated identifiers from crossing subsets.

### Grouped stratified split

The default subsets contain:

| Subset | Observations | Benign | Malignant | Purpose |
|---|---:|---:|---:|---|
| Training | 491 | 322 | 169 | Fit scaler and weights |
| Validation | 104 | 68 | 36 | Select iteration |
| Test | 104 | 68 | 36 | Final evaluation |

A deterministic greedy group assignment balances subset sizes and classes while treating each repeated sample code as indivisible.

### Training-only scaling

For feature $j$, the improved transformation is

$$
x_j'
=
\frac{x_j-\mu_{j,\mathrm{train}}}
{s_{j,\mathrm{train}}}.
$$

The validation and test values never contribute to $\mu_{j,\mathrm{train}}$ or $s_{j,\mathrm{train}}$. This removes preprocessing leakage and makes the evaluation boundary meaningful.

## Model and numerical improvements

### Stable logistic model

For a standardised vector $x$, the model score is

$$
z=w_0+x^Tw,
$$

and the malignant-class output is

$$
p(y=1\mid x)
=
\sigma(z)
=
\frac{1}{1+e^{-z}}.
$$

The stable sigmoid evaluates positive and negative scores through separate algebraic branches to avoid overflow.

### Stable loss

The improved objective is

$$
J(w)
=
\frac1N\sum_{i=1}^N
\left[
\log(1+e^{z_i})-y_i z_i
\right]
+
\frac{\lambda}{2}\sum_{j=1}^{d}w_j^2.
$$

The implementation evaluates

$$
\log(1+e^z)
$$

with `np.logaddexp(0, z)`. The bias is excluded from regularisation.

### Analytical gradient

With a design matrix $X$ that includes a bias column,

$$
\nabla J(w)
=
\frac1N X^T[\sigma(Xw)-y]
+
\lambda w_{\mathrm{nonbias}}.
$$

Replacing automatic differentiation reduces dependencies and makes the optimisation step explicit. A finite-difference test verifies the gradient numerically.

### Validation-based model selection

Training records loss and error on the training and validation subsets. The saved weights minimize validation loss. Training can stop when validation loss fails to improve by at least `1e-6` for 250 iterations.

This replaces the original selection rule

$$
k_\star
=
\operatorname*{arg\,min}_k
\text{training misclassifications}(k),
$$

which used the same observations for fitting and selection.

## Evaluation improvements

Malignant disease is the positive class. The rebuilt metrics include

$$
\mathrm{Precision}
=
\frac{TP}{TP+FP},
$$

$$
\mathrm{Recall}
=
\frac{TP}{TP+FN},
$$

$$
\mathrm{Specificity}
=
\frac{TN}{TN+FP},
$$

and

$$
\mathrm{Balanced\ accuracy}
=
\frac12
\left(
\mathrm{Recall}+\mathrm{Specificity}
\right).
$$

These metrics expose clinically relevant error types that overall accuracy alone can conceal.

## Measured impact

The selected default iteration was `4993`. The held-out test confusion matrix was:

| | Predicted malignant | Predicted benign |
|---|---:|---:|
| Actual malignant | 31 | 5 |
| Actual benign | 3 | 65 |

The corresponding results were:

| Metric | Result |
|---|---:|
| Accuracy | 92.31% |
| Malignant precision | 91.18% |
| Malignant recall | 86.11% |
| Benign specificity | 95.59% |
| Malignant F1 | 88.57% |
| Balanced accuracy | 90.85% |

These figures are less optimistic than evaluating on the training population, but they are substantially more credible because the 104 test observations did not influence scaling, fitting, early stopping, or threshold selection.

## Gate 3 repository impact

The Gate 3 archive includes:

- the rebuilt classifier and CLI;
- the unchanged raw UCI dataset and checksum;
- dataset attribution and transformation documentation;
- a non-destructive CSV preparation script;
- reference metrics and two verified figures;
- model and data-pipeline tests;
- a GitHub Actions workflow for Python 3.10 and 3.12;
- requirements and `.gitignore` files;
- `SAFETY.md` with explicit medical-use restrictions;
- a publication-quality `README.md` covering the data provenance, leakage controls, mathematical model, verified metrics, commands, outputs, tests, limitations, and safety boundary.

### Documentation impact

The Gate 3 README makes the repository independently understandable and reproducible. It presents the logistic-regression equations with rendered LaTeX, identifies the exact UCI source and eight-feature scope, explains the grouped split and train-only standardisation, reports the held-out confusion matrix and derived metrics, and directs readers to the medical-use restrictions. The documentation reduces the risk that the result is mistaken for a clinically validated model or that the holdout score is misread as a universal performance guarantee. This gate changed documentation only; it did not alter code, data, or measured results.

## Remaining limitations and safety impact

- This is an educational linear classifier, not a clinically validated system.
- The result comes from one deterministic holdout split.
- `Bare Nuclei` is omitted instead of imputed.
- The model probabilities are not calibrated for clinical risk.
- Synthetic vectors A, B, and C are demonstrations, not patient predictions.
- The repository must not be used for diagnosis, screening, treatment, triage, or medical decisions.
- No open-source licence was applied to the produced archive.
