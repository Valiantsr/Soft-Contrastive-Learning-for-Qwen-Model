from typing import List, Tuple, Dict, Optional
import torch, mteb
from mteb.evaluation.evaluators import BitextMiningEvaluator
from .imascl_mteb_adapter import IMASCLAdapter

@torch.no_grad()
def run_mteb_flores(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],      # contoh: [("ind_Latn","sun_Latn")]  — 1 arah per item
    device: Optional[str] = "cuda",
    max_length: int = 96,
    batch_size: int = 8,
    pooling: str = "mean",
    eval_splits = ("devtest",),
) -> Dict[str, float]:

    # Adapter + pad_token untuk decoder-only
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or tokenizer.unk_token
    adapter = IMASCLAdapter(model, tokenizer, device=device,
                            max_length=max_length, batch_size=batch_size, pooling=pooling)

    tasks = mteb.get_tasks(tasks=["FloresBitextMining"])
    if not tasks:
        raise RuntimeError("Task 'FloresBitextMining' tidak ditemukan di MTEB.")
    task = tasks[0] if not isinstance(tasks[0], type) else tasks[0]()
    task.load_data()

    flat: Dict[str, float] = {}
    langs_needed = sorted({x for (a, b) in pairs for x in (a, b)})

    for split in eval_splits:
        if split not in task.dataset:
            raise ValueError(f"Split '{split}' tidak ada. Tersedia: {list(task.dataset.keys())}")

        ds = task.dataset[split]
        print(f"[DBG] Split={split} → tipe={type(ds).__name__}")

        # Bangun {lang: list[str]}
        if hasattr(ds, "column_names"):
            missing = [l for l in langs_needed if l not in ds.column_names]
            if missing:
                raise ValueError(f"Kolom bahasa hilang di split '{split}': {missing}")
            sentences = {l: ds[l] for l in langs_needed}
        else:
            sentences = {l: ds[l] for l in langs_needed}

        n_full = len(next(iter(sentences.values())))
        print(f"[DBG] Dataset baris={n_full}; kolom aktif={list(sentences.keys())}")

        # Evaluasi SATU ARAH per item (hindari gold bentrok)
        for (src_lang, tgt_lang) in pairs:
            pair_columns = [(src_lang, tgt_lang)]
            print(f"[DBG] Evaluasi pair_columns={pair_columns} (target n={n_full})")

            evaluator = BitextMiningEvaluator(
                sentences,
                task_name="FloresBitextMining",
                pair_columns=pair_columns,
            )

            # >>> Paksa parameter internal evaluator agar tidak default ke 2
            evaluator.n = n_full
            # Gold untuk bitext paralel = pasangan diagonal (i→i)
            try:
                evaluator.gold = [(i, i) for i in range(n_full)]
            except Exception:
                pass

            scores = evaluator(adapter, encode_kwargs={"batch_size": batch_size})

            # ---- DEBUG: lihat bentuk mentah ----
            try:
                print(f"[DBG] Raw evaluator keys (top-level): {list(scores.keys())[:5]}")
            except Exception:
                pass
            
            # ---- Normalisasi bentuk keluaran evaluator ----
            if isinstance(scores, dict):
                if "scores" in scores and isinstance(scores["scores"], dict) and scores["scores"]:
                    pairs_scores = scores["scores"]          # bentuk A
                elif "languages" in scores and isinstance(scores["languages"], dict) and scores["languages"]:
                    pairs_scores = scores["languages"]       # bentuk B (jarang untuk evaluator langsung)
                else:
                    pairs_scores = scores                    # bentuk C: langsung per-pair di top-level
            else:
                raise ValueError(f"Unexpected evaluator output type: {type(scores)}")

            # Flatten
            # for subset_name, metrics in (scores.get("scores", {}) or {}).items():
            for subset_name, metrics in pairs_scores.items():
                for k, v in metrics.items():
                    flat[f"FloresBitextMining.{split}.{subset_name}.{k}"] = v
    return flat
