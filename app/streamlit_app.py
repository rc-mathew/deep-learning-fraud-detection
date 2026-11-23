"""Streamlit demo.  Run:  streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud_dl.serve import Scorer  # noqa: E402


@st.cache_resource
def load() -> Scorer:
    return Scorer()


st.set_page_config(page_title="Fraud scoring demo", layout="centered")
st.title("Deep learning fraud scoring")
st.caption("IEEE-CIS test month (unseen in training). The transformer runs live on CPU with ONNX Runtime; "
           "the ensemble score, reason codes and similar cases come from the week 8 run.")

scorer = load()
tx = scorer.tx
show = st.radio("Show", ["All", "Actual fraud", "Actual legitimate"], horizontal=True)
pool = tx if show == "All" else tx[tx["isFraud"] == (1 if show == "Actual fraud" else 0)]
label = {int(r.TransactionID): f"{int(r.TransactionID)}  ·  ${r.TransactionAmt:,.2f}  ·  {r.card4} {r.card6}"
         for r in pool.itertuples()}
tid = st.selectbox("Transaction", list(label), format_func=label.get)

r = scorer.score(tid)
c1, c2, c3 = st.columns(3)
c1.metric("Ensemble fraud probability", f"{r['ensemble_score']:.1%}")
c2.metric("Decision", r["decision"].upper())
c3.metric("Actual", r["actual_label"])
st.write(f"Blocked above {r['threshold']:.1%}: the threshold that minimised missed-fraud dollars plus "
         f"$10 per blocked good customer on the validation month.")

st.subheader("Why (top fields pushing towards fraud)")
st.write(", ".join(r["reasons"]) or "No field pushed towards fraud.")
st.caption("Captum Integrated Gradients on the transformer. IEEE-CIS anonymises what C and M fields mean.")

st.subheader("Most similar past fraud cases")
st.dataframe(pd.DataFrame(r["similar_past_fraud"]), hide_index=True)

st.caption(f"Transformer score {r['transformer_score']:.3f}, computed in {r['transformer_latency_ms']:.2f} ms. "
           "Code: github.com/rc-mathew/deep-learning-fraud-detection")
