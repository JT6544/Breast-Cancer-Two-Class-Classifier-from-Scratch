# Medical and modelling limitations

This repository is an educational machine-learning demonstration. It has not
been clinically validated and must not be used for diagnosis, screening,
treatment, triage, or any other medical decision.

The dataset is historical, small, and not representative of every population
or clinical setting. The model uses eight ordinal cytology features and omits
`Bare Nuclei` to preserve the scope of the original coursework experiment.

Reported probabilities are uncalibrated model outputs. They are not clinical
risk estimates. The synthetic feature vectors included in the program are
illustrations only and do not describe real patients.

The held-out test set is used once for educational evaluation. Its metrics do
not establish safety, fairness, generalisability, or clinical utility.
