"""A small fake dataset with IEEE-CIS's column layout, for tests only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd



def make_raw(n_cards: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    tid = 2_987_000
    for card in range(n_cards):
        card1 = int(rng.integers(1000, 1100))
        addr1 = float(rng.choice([100.0, 200.0, np.nan], p=[0.45, 0.45, 0.1]))
        first_day = int(rng.integers(0, 150))
        n = int(rng.integers(1, 25))
        times = np.sort(rng.integers(first_day * 86_400, 182 * 86_400, size=n))
        risky = rng.random() < 0.1
        for t in times:
            day = t // 86_400
            amt = float(np.round(rng.lognormal(4, 1) * (3 if risky else 1), 2))
            rows.append({
                "TransactionID": tid, "TransactionDT": int(t) + 86_400,
                "isFraud": int(rng.random() < (0.5 if risky else 0.02)),
                "TransactionAmt": amt, "ProductCD": rng.choice(list("WHCSR")),
                "card1": card1, "card2": float(rng.integers(100, 600)), "card3": 150.0,
                "card4": rng.choice(["visa", "mastercard"]), "card5": 226.0,
                "card6": rng.choice(["debit", "credit"]), "addr1": addr1, "addr2": 87.0,
                "dist1": float(rng.exponential(20)) if rng.random() < 0.6 else np.nan,
                "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan]),
                "R_emaildomain": rng.choice(["gmail.com", np.nan]),
                **{f"C{i}": float(rng.integers(0, 5)) for i in range(1, 15)},
                # D1 = days since the card's first transaction (+1 for the DT shift above).
                "D1": float(day - first_day + 1), "D4": float(rng.integers(0, 300)),
                "D10": float(rng.integers(0, 300)), "D15": float(rng.integers(0, 300)),
                **{f"M{i}": rng.choice(["T", "F", np.nan]) for i in range(1, 10)},
                **{f"V{i}": float(rng.random()) for i in range(1, 6)},
            })
            tid += 1
    txn = pd.DataFrame(rows)
    has_id = rng.random(len(txn)) < 0.3
    ident = pd.DataFrame({
        "TransactionID": txn.loc[has_id, "TransactionID"].to_numpy(),
        "id-30": rng.choice(["Windows 10", "iOS 11.2.1"], has_id.sum()),
        "id-31": rng.choice(["chrome 63.0", "mobile safari 11.0"], has_id.sum()),
        "id-33": rng.choice(["1920x1080", "2208x1242"], has_id.sum()),
        "DeviceType": rng.choice(["desktop", "mobile"], has_id.sum()),
        "DeviceInfo": rng.choice(["Windows", "iOS Device", "SM-G930V"], has_id.sum()),
    })
    return txn, ident


def write_raw(dir_path: Path, **kwargs) -> Path:
    txn, ident = make_raw(**kwargs)
    dir_path.mkdir(parents=True, exist_ok=True)
    txn.to_csv(dir_path / "train_transaction.csv", index=False)
    ident.to_csv(dir_path / "train_identity.csv", index=False)
    return dir_path

