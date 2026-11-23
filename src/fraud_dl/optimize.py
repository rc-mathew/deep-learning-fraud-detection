"""Week 8: ONNX export, INT8 quantisation and CPU latency for the transformer; FAISS similar-case search.

The laptop serves only these exported files: no PyTorch training stack needed at scoring time.
"""

from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .metrics import fraud_report
from .transformer import SeqData, TxnTransformer

INPUTS = ["cat", "num", "isna", "pad"]


def export_onnx(model: TxnTransformer, data: SeqData, path: Path) -> Path:
    model = model.cpu().eval()
    example = tuple(t.cpu() for t in data.batch(np.arange(2)))
    torch.onnx.export(model, example, str(path), input_names=INPUTS, output_names=["logit"],
                      dynamic_axes={n: {0: "batch"} for n in INPUTS + ["logit"]}, opset_version=17,
                      dynamo=False)
    return path


def quantize(fp32: Path, int8: Path) -> Path:
    """Dynamic INT8: weights stored as 8-bit integers, activations quantised on the fly.
    Gather is included because the embedding tables are most of this model's size."""
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm", "Gather"])
    return int8


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # one scoring request per core, as a small API would run
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_scores(sess: ort.InferenceSession, data: SeqData, rows: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    out = []
    for i in range(0, len(rows), batch_size):
        feed = {n: t.cpu().numpy() for n, t in zip(INPUTS, data.batch(rows[i:i + batch_size]))}
        out.append(1 / (1 + np.exp(-sess.run(None, feed)[0])))
    return np.concatenate(out)


def latency_ms(fn, data: SeqData, rows: np.ndarray, n: int = 300) -> dict[str, float]:
    """Single-transaction scoring latency on this machine's CPU, including window building."""
    times = []
    for r in rows[:n]:
        t0 = time.perf_counter()
        fn(np.array([r]))
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def compare(model: TxnTransformer, data: SeqData, out_dir: Path, test: np.ndarray) -> dict:
    """fp32 PyTorch vs fp32 ONNX vs INT8 ONNX: size, latency and test accuracy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = export_onnx(model, data, out_dir / "transformer_fp32.onnx")
    int8 = quantize(fp32, out_dir / "transformer_int8.onnx")
    model = model.cpu().eval()
    torch.set_num_threads(1)

    @torch.no_grad()
    def torch_fn(rows):
        return torch.sigmoid(model(*[t.cpu() for t in data.batch(rows)])).numpy()

    res = {}
    for name, fn, size in [
        ("pytorch_fp32", torch_fn, sum(p.numel() * 4 for p in model.parameters())),
        ("onnx_fp32", lambda r, s=session(fp32): onnx_scores(s, data, r), fp32.stat().st_size),
        ("onnx_int8", lambda r, s=session(int8): onnx_scores(s, data, r), int8.stat().st_size),
    ]:
        scores = np.concatenate([fn(test[i:i + 4096]) for i in range(0, len(test), 4096)])
        rep = fraud_report(data.y[test], scores)
        res[name] = {"size_mb": round(size / 1e6, 3), **latency_ms(fn, data, test),
                     "recall_at_1pct_fpr": rep["recall_at_1pct_fpr"], "pr_auc": rep["pr_auc"]}
        res[name]["_scores"] = scores
    diff = np.abs(res["onnx_int8"]["_scores"] - res["pytorch_fp32"]["_scores"])
    res["int8_max_score_diff"], res["int8_mean_score_diff"] = float(diff.max()), float(diff.mean())
    for k in ("pytorch_fp32", "onnx_fp32", "onnx_int8"):
        res[k].pop("_scores")
    return res


@torch.no_grad()
def case_embeddings(model: TxnTransformer, data: SeqData, rows: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """The transformer's view of each transaction (its last-token vector), L2-normalised."""
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        h, _ = model.encode(*data.batch(rows[i:i + batch_size]))
        out.append(h[:, -1].float().cpu().numpy())
    e = np.concatenate(out).astype(np.float32)
    faiss.normalize_L2(e)
    return e


class CaseIndex:
    """Past confirmed fraud cases, searchable by cosine similarity: "have we seen this before?"."""

    def __init__(self, embeddings: np.ndarray, transaction_ids: np.ndarray):
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.ids = np.asarray(transaction_ids)

    def search(self, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        sim, idx = self.index.search(queries, k)
        return self.ids[idx], sim

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))
        np.save(path.with_suffix(".ids.npy"), self.ids)

    @classmethod
    def load(cls, path: Path) -> "CaseIndex":
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path))
        obj.ids = np.load(path.with_suffix(".ids.npy"))
        return obj
