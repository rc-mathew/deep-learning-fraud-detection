# Deep Learning Fraud Detection

Fraud detection on [IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection) built from deep learning models:
a transaction transformer with self-supervised pretraining, a graph neural network over shared
devices and emails, and an autoencoder for unlabelled anomalies. XGBoost is kept as one baseline row.

Every model is judged on the same **time-based split** (train on the past, test on the last 30 days) and on
**recall at 1% false-positive rate**, not ROC-AUC alone.

## Results (test set: last 30 days)

| Model | Recall @ 1% FPR | Recall @ 5% FPR | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- |
| XGBoost baseline | _week 4_ | | | |
| Transformer, from scratch | _week 6_ | | | |
| Transformer, pretrained | _week 6_ | | | |
| GNN (GraphSAGE) | _week 7_ | | | |
| Autoencoder | _week 7_ | | | |
| Ensemble | _week 8_ | | | |

## Data pipeline (week 4)

1. **Load** `train_transaction.csv` + `train_identity.csv` (Kaggle's test files have no labels, so they are not used).
2. **Card id.** IEEE-CIS has no card id. `uid = card1 + addr1 + (day − D1)`, since D1 counts days since
   the card's first transaction.
3. **Time split.** Last 30 days = test, the 30 before = validation, the rest = train
   (`config.GAP_DAYS` adds a gap to mimic late chargebacks).
4. **Velocity features**, past-only per card: transaction counts and amount sums in the last 1 h / 24 h / 7 d,
   seconds since the previous transaction, and amount vs the card's earlier average.
5. **Sequence table** for the transformer: one encoded row per transaction, sorted by card then time.
   Vocabularies and scaling are fitted on train rows only. `window_indices()` returns each transaction's
   last 32 card transactions. Earlier transactions' labels are never inputs.

```
src/fraud_dl/
  config.py      split sizes, windows, sequence columns
  data.py        load, card id, time split, sort by card
  features.py    velocity features
  sequences.py   encoder + sequence windows
  metrics.py     recall @ FPR, PR-AUC, ROC-AUC
  prepare.py     python -m fraud_dl.prepare --data-dir ... --out-dir artifacts
  baseline.py    python -m fraud_dl.baseline --artifacts artifacts
notebooks/week4_data_and_baseline.ipynb   runs both on Kaggle or Colab
tests/           pipeline tests on a synthetic IEEE-CIS-shaped dataset
```

## Run

On Kaggle or Colab, open `notebooks/week4_data_and_baseline.ipynb` (setup steps are in its first cell). Locally:

```bash
pip install -e ".[dev]"
pytest
python -m fraud_dl.prepare --data-dir path/to/ieee-fraud-detection --out-dir artifacts
python -m fraud_dl.baseline --artifacts artifacts
```

## Known caveat

IEEE-CIS labels later transactions of a defrauded account as fraud too. Models that see card history can
pick that up, which flatters them compared with production. `data_report.json` measures how strong the effect is.
