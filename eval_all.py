# eval_all.py
from typing import Dict, List, Optional, Tuple
import numpy as np

# asumsi kamu sudah punya modul ini (punya run_eval yang return dict {'avg_cosine':..., 'retrieval': {...}})
import evaluate.eval_sts_retrieval as ev
from evaluate.eval_mteb import run_mteb_flores

def run_hits_cosine(
    src_texts: List[str],
    tgt_texts: List[str],
    tokenizer,
    model,
    device: str = "cuda",
    ks = (1, 5, 10),
    max_length: int = 128,
    batch_size: int = 32,
    layer: int = -1,
    pooling: str = "mean",
) -> Dict:
    # Jika eval_sts_retrieval.batch_encode belum punya 'pooling', kamu bisa fork pendek:
    # - atau tetap pakai versi lama; kunci-nya konsisten train↔eval.
    res = ev.run_eval(
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        labels=None,
        tokenizer=tokenizer,
        model=model,
        device=device,
        retrieval=True,
        error_analysis_k=5,
    )
    # Pastikan kunci keluarannya konsisten:
    out = {
        "avg_cosine": float(res.get("avg_cosine", np.nan)),
    }
    if "retrieval" in res and isinstance(res["retrieval"], dict):
        for k, v in res["retrieval"].items():
            out[f"hits@{k.split('@')[-1]}"] = float(v)
    return out

def run_all_eval(
    src_texts: List[str],
    tgt_texts: List[str],
    tokenizer,
    model,
    *,
    device: str = "cuda",
    ks = (1, 5, 10),
    pooling: str = "mean",
    max_length: int = 128,
    batch_size: int = 32,
    mteb_languages: Optional[List[Tuple[str, str]]] = None,
    mteb_output_folder: str = "mteb_results/imascl_flores",
) -> Dict:
    report = {}
    # 1) evaluasi internal (cosine + hits)
    internal = run_hits_cosine(
        src_texts, tgt_texts, tokenizer, model,
        device=device, ks=ks, max_length=max_length, batch_size=batch_size, pooling=pooling
    )
    report.update({f"internal.{k}": v for k, v in internal.items()})
    # 2) evaluasi MTEB (opsional)
    if mteb_languages:
        mteb_res = run_mteb_flores(
            model=model,
            tokenizer=tokenizer,
            languages=mteb_languages,
            device=device,
            max_length=max_length,
            batch_size=max_length if batch_size is None else batch_size,  # bebas diubah
            pooling=pooling,
            output_folder=mteb_output_folder,
            eval_splits=("test",),
        )
        # gabung
        for k, v in mteb_res.items():
            report[f"mteb.{k}"] = v

    return report
