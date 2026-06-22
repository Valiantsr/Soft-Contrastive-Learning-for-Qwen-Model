import numpy as np
import torch
from typing import List, Optional, Dict, Tuple

# ---------- Core utils ----------
def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

def _encode(
    texts: List[str],
    tokenizer,
    model,
    device: str = "cuda",
    max_length: int = 128,
    layer: int = -1,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    model.config.output_hidden_states = True
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length).to(device)
            hs = model(**inputs).hidden_states[layer]               # (B,T,H)
            mask = inputs["attention_mask"].unsqueeze(-1).float()   # (B,T,1)
            sent = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embs.append(sent.cpu())
    if not embs:
        raise RuntimeError("Tidak ada embedding yang dihasilkan.")
    return torch.cat(embs, dim=0).numpy()

def _hits_at_k(
    q: np.ndarray, c: np.ndarray, ks=(1,), gold: Optional[np.ndarray] = None
) -> Dict[str, float]:
    q, c = _l2(q), _l2(c)
    sim = q @ c.T
    if gold is None:                       # diagonal mode (pairwise 1–1)
        N = sim.shape[0]
        assert sim.shape == (N, N), "Sim harus NxN untuk diagonal."
        gold = np.arange(N)
    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        hits = (order[:, :k] == gold[:, None]).any(axis=1)
        out[f"accuracy@{k}"] = float(hits.mean())
    return out

# ---------- High-level API ----------
def run_eval(
    src_texts: List[str],
    tgt_texts: List[str],
    labels: Optional[List[float]],
    tokenizer,
    model,
    device: str = "cuda",
    mode: str = "pairwise",                # "pairwise" | "triplet"
    negatives: Optional[List[str]] = None, # untuk mode triplet
    ks: Tuple[int, ...] = (1,),
    max_length: int = 128,
    layer: int = -1,
    batch_size: int = 32,
    verbose: bool = True,
    retrieval: bool = True,                # kompatibilitas
    **kwargs,
) -> Dict[str, float]:

    if mode == "triplet":
        # anchors = src_texts, positives = tgt_texts, negatives opsional
        assert len(src_texts) == len(tgt_texts), "anchors dan positives harus sama panjang."
        cand_texts = list(dict.fromkeys(tgt_texts + (negatives or [])))  # unique & stable
        idx = {t: i for i, t in enumerate(cand_texts)}
        gold = np.array([idx[p] for p in tgt_texts], dtype=np.int64)

        emb_q   = _encode(src_texts, tokenizer, model, device, max_length, layer, batch_size)
        emb_pos = _encode(tgt_texts, tokenizer, model, device, max_length, layer, batch_size)
        emb_c   = _encode(cand_texts, tokenizer, model, device, max_length, layer, batch_size)

        # cosine anchor–positive
        qn, pn = _l2(emb_q), _l2(emb_pos)
        avg_cos_pos = float(np.sum(qn * pn, axis=1).mean())
        if verbose: print(f"\nAverage Cosine (anchor–positive): {avg_cos_pos:.4f}")

        # optional margin cos(ap) - cos(an) jika negatives tersedia & sejajar
        out: Dict[str, float] = {"avg_cosine_pos": avg_cos_pos}
        if negatives is not None and len(negatives) == len(src_texts):
            emb_neg = _encode(negatives, tokenizer, model, device, max_length, layer, batch_size)
            nn = _l2(emb_neg)
            margin = np.sum(qn * pn, axis=1) - np.sum(qn * nn, axis=1)
            out["avg_triplet_margin"] = float(margin.mean())
            if verbose: print(f"Average Triplet Margin (cos(ap)-cos(an)): {out['avg_triplet_margin']:.4f}")

        if retrieval:
            acc = _hits_at_k(emb_q, emb_c, ks=ks, gold=gold)
            if verbose:
                for k, v in acc.items(): print(f"{k}: {v:.4f}")
            out.update(acc)
        return out

    # ---- pairwise (SCL / NT-Xent) ----
    emb_src = _encode(src_texts, tokenizer, model, device, max_length, layer, batch_size)
    emb_tgt = _encode(tgt_texts, tokenizer, model, device, max_length, layer, batch_size)

    # cosine diagonal
    sn, tn = _l2(emb_src), _l2(emb_tgt)
    avg_cos = float(np.sum(sn * tn, axis=1).mean())
    if verbose: print(f"\nAverage Cosine (diag): {avg_cos:.4f}")

    out = {"avg_cosine": avg_cos}
    if retrieval:
        acc = _hits_at_k(emb_src, emb_tgt, ks=ks, gold=None)  # diagonal
        if verbose:
            for k, v in acc.items(): print(f"{k}: {v:.4f}")
        out.update(acc)
    return out
