# Digital Banking Fraud Risk and Customer Experience

QM640 Data Analytics Capstone, Walsh College.

This repository holds the data, documentation and code for a study of fraud exposure,
financial loss, false-positive fraud interventions and risk segmentation among U.S. bank
and credit union account holders.

## Research questions

| RQ | Question | Outcome | Method |
|---|---|---|---|
| RQ1 | Are mobile, online, P2P and automated-chat use associated with fraud experience, after controlling for customer characteristics? | `FRAUD_EXP` | Chi-square tests, logistic regression |
| RQ2 | Among fraud victims, are the fraud channel, scam type, bank detection and repeat victimization associated with money being taken? | `FIN_LOSS` | Chi-square / Fisher tests, logistic regression |
| RQ3 | Are digital-banking use and institution type associated with false-positive fraud interventions, after controlling for fraud experience? | `FALSE_POS` | Chi-square tests, logistic regression |
| RQ4 | Can machine-learning models predict fraud experience well enough to form risk segments? | `FRAUD_EXP` | Logistic regression, Random Forest, Gradient Boosting |

## Data

- Source: Consumer Financial Protection Bureau, National Age-Friendly Banking Survey (NAFBS) Public Use File.
- Fielded January 9 to February 9, 2024; 2,721 respondents; 284 variables.
- The labeled CSV is included in `data/raw/`. It is a public release by a U.S. federal agency and
  contains no personally identifying information. Official download links are in `docs/source_links.md`.
- The data dictionary for every variable used in the study is in `docs/data_dictionary.csv`.

## Repository structure

```
qm640-digital-banking-fraud-risk/
├── README.md
├── requirements.txt
├── data/
│   └── raw/cfpb_nafbs-puf-labeled_2026-02.csv
├── docs/
│   ├── data_dictionary.csv
│   └── source_links.md
├── src/
│   ├── prep.py               derivation rules for all variables
│   ├── 01_sample_size.py     sample-size calculations (synopsis Tables 4, 5 and A1)
│   └── 02_descriptives.py    sample profile and outcome prevalence (Table 6, Figure 3)
├── notebooks/                analysis notebooks for RQ1 to RQ4 (added during the project)
└── outputs/                  result tables and figures
```

## How to run

Requires Python 3.11 or later.

```bash
git clone <this repository URL>
cd qm640-digital-banking-fraud-risk
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/01_sample_size.py     # prints every sample-size input and result
python src/02_descriptives.py    # writes Table 6 and Figure 3 to outputs/
```

## Key derivation rules (`src/prep.py`)

| Variable | Rule |
|---|---|
| `FRAUD_EXP` | 1 if `FRAUD1` is one time or more, 0 if never; "I'm not sure" and skipped are missing |
| `FIN_LOSS` | 1 if `FRAUD2` reports money lost (recovered or not), 0 if no money was taken; fraud victims only |
| `UNREC_LOSS` | 1 if some or none of the loss was recovered, 0 otherwise; fraud victims only |
| `FALSE_POS` | 1 if `FRAUD8` is one time or more, 0 if never; skipped is missing |

Items about the fraud incident (`FRAUD2` to `FRAUD6`, `SCAM_*`) were asked only of fraud victims
and are used only in RQ2.

## Sample size

The final sample size is the largest minimum requirement across RQ1 to RQ4. Running
`src/01_sample_size.py` reproduces it and saves the detail to `outputs/sample_size_results.json`.

## Limitations

Survey data are self-reported and cross-sectional. Results are associations and predictive accuracy,
not causal effects, and the RQ4 model describes customer risk profiles, not a fraud detection system.
