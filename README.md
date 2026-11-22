# Deep Learning Fraud Detection

Fraud detection on [IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection) built from deep learning models:
a transaction transformer with self-supervised pretraining, a graph neural network over shared
devices and emails, and an autoencoder for unlabelled anomalies. XGBoost is kept as one baseline row.

Every model is judged on the same **time-based split** (train on the past, test on the last 30 days) and on
**recall at 1% false-positive rate**, not ROC-AUC alone.

## Results (test set: last 30 days)

| Model | Recall @ 1% FPR | Recall @ 5% FPR | PR-AUC | ROC-AUC | First-fraud recall @ 1% FPR |
| --- | --- | --- | --- | --- | --- |
| XGBoost baseline | **0.466** | **0.652** | **0.552** | **0.903** | **0.545** |
| Transformer, from scratch | 0.378 | 0.590 | 0.430 | 0.858 | 0.414 |
| Transformer, pretrained | 0.381 | 0.565 | 0.418 | 0.859 | 0.393 |
| GNN (GraphSAGE) | _week 7_ | | | | |
| Autoencoder | _week 7_ | | | | |
| Ensemble | _week 8_ | | | | |

First-fraud recall drops transactions from cards that already had a fraud (see Known caveat).

**Weeks 5-6 finding.** Masked-field pretraining worked as a task: its loss fell every epoch, from 1.24 to 0.44
over 10 epochs. It lifted validation PR-AUC from 0.515 to 0.557, but that gain did not carry to the test
month (0.430 vs 0.418). Pretraining saw the validation period's transactions, so its validation gain is
partly familiarity with that month. Both transformers trail XGBoost by about 9 points of recall at 1% FPR.
They see about 40 fields per transaction, while XGBoost uses all 434, including the V and full C/D columns.
Whether the transformer adds anything is tested in the week 8 ensemble.

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
   last 16 card transactions. Earlier transactions' labels are never inputs.

```
src/fraud_dl/
  config.py      split sizes, windows, sequence columns
  data.py        load, card id, time split, sort by card
  features.py    velocity features
  sequences.py   encoder + sequence windows
  metrics.py     recall @ FPR, PR-AUC, ROC-AUC
  prepare.py     python -m fraud_dl.prepare --data-dir ... --out-dir artifacts
  baseline.py    python -m fraud_dl.baseline --artifacts artifacts
  transformer.py transformer written from scratch, masked-field pretraining, fine-tuning
  transformer_run.py  python -m fraud_dl.transformer_run --artifacts artifacts
notebooks/week4_data_and_baseline.ipynb   week 4 on Kaggle or Colab
notebooks/week5_6_transformer.ipynb       weeks 5-6 on a Kaggle GPU
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

IEEE-CIS labels later transactions of a defrauded account as fraud too: after a card's first fraud, 44.7% of its
later transactions are fraud, against 3.5% overall. Models that see card history pick that up, which flatters
them compared with production. Every model therefore also reports **first-fraud recall**: recall at 1% FPR
on transactions from cards with no earlier fraud.
