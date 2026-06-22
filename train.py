# ============================================================
# train.py (versi final – decoder-only, single GPU, FLORES+)
# ============================================================
import argparse, os
import torch
import unicodedata
import re
import numpy as np
import modules.AMEDataCollator as dc_module
import importlib
from evaluate import eval_mteb
importlib.reload(eval_mteb)
from datasets                     import load_dataset, concatenate_datasets, DatasetDict, Dataset
from transformers                 import AutoConfig, AutoTokenizer,AutoModelForCausalLM
from modules.AMETrainer           import AMETrainer
from modules.AMEDataCollator      import AMEDataCollator
from modules.AMETrainingArguments import AMETrainingArguments
from modules.AMEMetric            import AMEMetric, RemoveBadSaveCallback, VerboseEvalCallback
from modules.BLEUMetric           import BLEUMetric
from modules.AMESampler           import LengthInBatchSampler
from evaluate.eval_sts_retrieval  import run_eval
# from evaluate.eval_all            import run_all_eval
from evaluate.eval_mteb           import run_mteb_flores
from huggingface_hub              import HfFileSystem

torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', default='jv_su_alignment')
    p.add_argument('--output', default='./output')
    p.add_argument('--log_dir', default='./logs')
    p.add_argument('--continue_from_last_ckpt', action='store_true')
    p.add_argument('--dataset_name', default='openlanguagedata/flores_plus')
    p.add_argument('--pairs', nargs='+', default=['jav_Latn-ind_Latn', 'sun_Latn-ind_Latn'])
    p.add_argument('--split', default='dev')
    p.add_argument('--model', default='Qwen/Qwen1.5-0.5B')
    p.add_argument('--embed_model', default=None)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--epoch', type=int, default=3)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--margin', type=float, default=0.0)
    p.add_argument('--positive_threshold', type=float, default=0.95)
    p.add_argument('--tau', type=float, default=0.1)
    p.add_argument('--w_func', default='raw', choices=['raw', 'Softmax'])
    p.add_argument('--train_mono_space', action='store_true')
    p.add_argument("--teacher_model_name_or_path", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--embed_direction", type=str, choices=["left", "right", "both", "input"], default="both")
    p.add_argument('--eval_steps', type=int, default=500)
    p.add_argument('--max_gen_length', type=int, default=None)
    p.add_argument('--loss_type', choices=["scl", "nt_xent", "triplet"], default="scl")
    p.add_argument('--label_mode', choices=["priority", "average"], default="priority")
    args = p.parse_args()
    args.world_size = 1
    args.per_device_batch_size = args.batch
    return args

# ------------------------------------------------------------------
def load_one_pair(dataset_name: str, src_cfg: str, tgt_cfg: str, split: str = "dev") -> list[dict]:
    ds_src = load_dataset(dataset_name, src_cfg, split=split)
    ds_tgt = load_dataset(dataset_name, tgt_cfg, split=split)
    assert len(ds_src) == len(ds_tgt)
    out, guid = [], 0
    for ex_s, ex_t in zip(ds_src, ds_tgt):
        out.append({"sentence1": ex_s["text"], "sentence2": ex_t["text"], "lang1": src_cfg.split("_")[0], "lang2": tgt_cfg.split("_")[0], "guid": guid})
        guid += 1
    return out

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,;:!?'-]", "", text)
    return text.strip()

def deduplicate_parallel(data):
    seen = set()
    unique = []
    for ex in data:
        # Hash kombinasi bahasa dan dua kalimat (biar paralel unik)
        h = (ex['lang1'], ex['lang2'], ex['sentence1'], ex['sentence2'])
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique

import random

def build_triplet_dataset(pairs, langs, num_triplet_per_pair=1):
    triplets = []
    n = len(pairs)
    for idx, (anchor, positive) in enumerate(pairs):
        anchor_lang1, anchor_lang2 = langs[idx]
        for _ in range(num_triplet_per_pair):
            neg_idx = random.choice([i for i in range(n) if i != idx])
            negative = pairs[neg_idx][1]  # gunakan sisi "positive" dari pasangan lain sebagai negative
            # neg_lang2 = langs[neg_idx][1]
            # triplets.append((anchor, positive, negative))
            triplets.append({
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "lang1": anchor_lang1,
                "lang2": anchor_lang2,   # atau "lang_pos": lang2
            #     "lang_neg": neg_lang2  # opsional, jika butuh
            })
    return triplets

def build_parallel_dataset(args, split="dev", loss_type="scl", num_triplet_per_pair=1):
    merged: list[dict] = []
    global_guid = 0
    for pair in args.pairs:
        src_cfg, tgt_cfg = pair.split('-')
        ds_src = load_dataset(args.dataset_name, src_cfg, split=split)
        ds_tgt = load_dataset(args.dataset_name, tgt_cfg, split=split)
        assert len(ds_src) == len(ds_tgt), f"Mismatch data size: {src_cfg} vs {tgt_cfg}"
        for ex_s, ex_t in zip(ds_src, ds_tgt):
            sent1 = normalize_text(ex_s["text"])
            sent2 = normalize_text(ex_t["text"])
            if not (4 <= len(sent1.split()) <= 128): continue
            if not (4 <= len(sent2.split()) <= 128): continue
            merged.append({
                "sentence1": sent1,
                "sentence2": sent2,
                "lang1":     src_cfg.split("_")[0],
                "lang2":     tgt_cfg.split("_")[0],
                "guid":      global_guid
            })
            global_guid += 1
    merged = deduplicate_parallel(merged)
    if loss_type == "triplet":
        pairs = [(ex["sentence1"], ex["sentence2"]) for ex in merged]
        langs = [(ex["lang1"], ex["lang2"]) for ex in merged]
        triplets = build_triplet_dataset(pairs, langs, num_triplet_per_pair=num_triplet_per_pair)
        triplet_dicts = triplets
        ds_triplet = Dataset.from_list(triplet_dicts)
        if split == "dev":
            train_size = int(0.9 * len(ds_triplet))
            valid_size = len(ds_triplet) - train_size
            return DatasetDict({
                "train": ds_triplet.select(range(train_size)),
                "valid": ds_triplet.select(range(train_size, len(ds_triplet))),
            })
        else:
            return DatasetDict({
                "test": ds_triplet,
            })
    else:
        ds_full = Dataset.from_list(merged)
        ds_full = ds_full.shuffle(seed=42)
        total = len(ds_full)
        if split == "dev":
            train_size = int(0.9 * total)
            valid_size = total - train_size
            return DatasetDict({
                "train": ds_full.select(range(train_size)),
                "valid": ds_full.select(range(train_size, total)),
            })
        else:  # For devtest/test: do not split, just return all as 'test'
            return DatasetDict({
                "test": ds_full,
            })
# ------------------------------------------------------------------
def build_tokenizer_and_collator(args):
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = 'left'
    tok.truncation_side = 'right'
    tok.model_max_length = 512
    if tok.pad_token is None or tok.pad_token_id >= tok.vocab_size:
        # tok.pad_token = tok.eos_token
        tok.add_special_tokens({"pad_token": "[PAD]"})

    if hasattr(args, "loss_type") and args.loss_type == "triplet":
        # Kembalikan tokenizer dan collator triplet (collate_fn_triplet)
        def collate_fn_triplet(batch):
            anchors = [item["anchor_input_ids"] for item in batch]
            anchor_masks = [item["anchor_attention_mask"] for item in batch]
            positives = [item["positive_input_ids"] for item in batch]
            positive_masks = [item["positive_attention_mask"] for item in batch]
            negatives = [item["negative_input_ids"] for item in batch]
            negative_masks = [item["negative_attention_mask"] for item in batch]
            # Padding
            from torch.nn.utils.rnn import pad_sequence
            import torch
            def pad(lst, pad_val):
                return pad_sequence([torch.tensor(x, dtype=torch.long) for x in lst],
                                   batch_first=True, padding_value=pad_val)
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            batch_out = {
                "anchor_input_ids": pad(anchors, pad_id),
                "anchor_attention_mask": pad(anchor_masks, 0),
                "positive_input_ids": pad(positives, pad_id),
                "positive_attention_mask": pad(positive_masks, 0),
                "negative_input_ids": pad(negatives, pad_id),
                "negative_attention_mask": pad(negative_masks, 0),
            }
            return batch_out
        return tok, collate_fn_triplet
    
    dc_module.teacher_tokenizer = tok
    collator = AMEDataCollator(
        tokenizer        = tok,
        pad_to_multiple_of = 8,
        device           = None,
        embed_direction  = args.embed_direction
    )
    return tok, collator

def mk_tokenize_fn(student_tok, teacher_tok, args):
    def _tok(ex):
        if "anchor" in ex and "positive" in ex and "negative" in ex:
            out = {}
            # Tokenize anchor
            anchor = student_tok(ex["anchor"], max_length=512, truncation=True)
            out["anchor_input_ids"] = anchor["input_ids"]
            out["anchor_attention_mask"] = anchor["attention_mask"]
            # Tokenize positive
            positive = student_tok(ex["positive"], max_length=512, truncation=True)
            out["positive_input_ids"] = positive["input_ids"]
            out["positive_attention_mask"] = positive["attention_mask"]
            # Tokenize negative
            negative = student_tok(ex["negative"], max_length=512, truncation=True)
            out["negative_input_ids"] = negative["input_ids"]
            out["negative_attention_mask"] = negative["attention_mask"]
            return out
        
        src_text = ex['sentence1']
        tgt_text = ex['sentence2']
        src = student_tok(src_text, max_length=512, truncation=True)
        tgt = student_tok(tgt_text, max_length=512, truncation=True)
        out = {
            'input_ids_1': src['input_ids'], 'attention_mask_1': src['attention_mask'],
            'input_ids_2': tgt['input_ids'], 'attention_mask_2': tgt['attention_mask'],
            'input_ids': src['input_ids'],
        }
        if teacher_tok is not None:
            if args.embed_direction in ("left", "both", "input"):
                t_src = teacher_tok(src_text, max_length=512, truncation=True)
                out.update({ 'embed_input_ids_1': t_src['input_ids'], 'embed_attention_mask_1': t_src['attention_mask'] })
            if args.embed_direction in ("right", "both"):
                t_tgt = teacher_tok(tgt_text, max_length=512, truncation=True)
                out.update({ 'embed_input_ids_2': t_tgt['input_ids'], 'embed_attention_mask_2': t_tgt['attention_mask'] })
        return out
    return _tok
    
# ------------------------------------------------------------------
def combined_metrics(pred):
    results = {}
    alignment_out, gen_out = pred.predictions
    class P1: pass
    P1.predictions = alignment_out
    results.update( AMEMetric()(P1) )
    if gen_out is not None and hasattr(pred, "label_ids"):
        class P2: pass
        P2.predictions = gen_out
        P2.label_ids   = pred.label_ids
        P2.tokenizer   = tokenizer  # pas inisiasi trainer, bawa tokenizer sini
        results.update( BLEUMetric(tokenizer)(P2) )
    return results
# ------------------------------------------------------------------
def prepare_trainer(args, with_test=False):
    ds_dict_dev = build_parallel_dataset(args, split="dev")
    student_tokenizer, collator = build_tokenizer_and_collator(args)
    teacher_tokenizer = None
    if args.teacher_model_name_or_path is not None:
        from transformers import AutoTokenizer
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name_or_path)
    tok_fn = mk_tokenize_fn(student_tokenizer, teacher_tokenizer, args)
    ds_train = ds_dict_dev['train'].map(tok_fn, batched=True, num_proc=4)
    ds_valid = ds_dict_dev['valid'].map(tok_fn, batched=True, num_proc=4)
    ds_test = None
    if with_test:
        ds_dict_test = build_parallel_dataset(args, split="devtest")
        ds_test = ds_dict_test["test"].map(tok_fn, batched=True, num_proc=4)

    train_sampler = LengthInBatchSampler(
        dataset=ds_train,
        batch_size=args.batch,
        seed=42,
        shuffle=True,)

    eval_sampler = LengthInBatchSampler(
        dataset=ds_valid,
        batch_size=args.batch,
        seed=42,
        shuffle=False,  # biasanya eval/test tidak shuffle
    )
    test_sampler = None
    if ds_test is not None:
        test_sampler = LengthInBatchSampler(
            dataset=ds_test,
            batch_size=args.batch,
            seed=42,
            shuffle=False,
        )

    config = AutoConfig.from_pretrained(args.model, output_hidden_states=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config).to(device)
    model.resize_token_embeddings(len(student_tokenizer))

    teacher_model = None
    if args.teacher_model_name_or_path is not None:
        from transformers import AutoModel
        print(f"[DEBUG] Loading teacher model from: {args.teacher_model_name_or_path}")
        teacher_model = AutoModel.from_pretrained(
            args.teacher_model_name_or_path,
            output_hidden_states=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        teacher_model.resize_token_embeddings(len(student_tokenizer))
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.eval()
        for name, p in teacher_model.named_parameters():
            if p.requires_grad:
                raise RuntimeError(f"⚠️ Teacher parameter {name} masih requires_grad=True!")
        print("✅ Semua parameter teacher telah di‐freeze (requires_grad=False)")

    training_args = AMETrainingArguments(
        output_dir=args.output,
        run_name=args.run_name,
        logging_dir=args.log_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        fp16=True,
        tf32=True,
        lr_scheduler_type="linear",
        warmup_steps=500,
        max_grad_norm=1.0,
        save_strategy="steps",
        evaluation_strategy="steps",
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        remove_unused_columns=False,
        group_by_length=True,
        gradient_accumulation_steps=2,
        margin=args.margin,
        positive_threshold=args.positive_threshold,
        tau=args.tau,
        w_func=args.w_func,
        embed_model_dir=args.embed_model,
        embed_direction=args.embed_direction,
        train_mono_space=args.train_mono_space,
        # mono_weight=args.mono_weight,
        # mono_tau=args.mono_tau,
        # mono_type=args.mono_type,
        small_batch_size=None,
        ddp_find_unused_parameters=False,
        metrics_for_save_model=("eval_loss",),
        metrics_larger_better=(False,),
        seed=42,
        prediction_loss_only=False,
        predict_with_generate=True,
        generation_max_length=128,
        loss_type=None,
        label_mode=None,
        report_to="wandb",
    )
    return model, training_args, ds_train, ds_valid, ds_test, collator, teacher_model, train_sampler, eval_sampler, student_tokenizer

# ------------------------------------------------------------------
def get_embedding_scores(model, tokenizer, src_texts, tgt_texts, device="cuda", batch_size=16):
    model.eval()
    all_sims = []
    with torch.no_grad():
        for i in range(0, len(src_texts), batch_size):
            batch_src = src_texts[i:i+batch_size]
            batch_tgt = tgt_texts[i:i+batch_size]
            inputs_src = tokenizer(batch_src, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            inputs_tgt = tokenizer(batch_tgt, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            out_src = model(**inputs_src, output_hidden_states=True, return_dict=True)
            out_tgt = model(**inputs_tgt, output_hidden_states=True, return_dict=True)
            
            emb_src = out_src.hidden_states[-1].mean(dim=1)
            emb_tgt = out_tgt.hidden_states[-1].mean(dim=1)
            
            sims = torch.nn.functional.cosine_similarity(emb_src, emb_tgt, dim=-1)
            all_sims.extend(sims.detach().cpu().numpy())
    return np.array(all_sims)
    
# ------------------------------------------------------------------
def main():
    args = parse()
    model, tr_args, ds_train, ds_valid, ds_test, collator, teacher_model, train_sampler, eval_sampler, student_tokenizer = prepare_trainer(args, with_test=True)
    trainer = AMETrainer(
        model=model,
        args=tr_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=collator,
        compute_metrics=AMEMetric(),
        teacher_model=teacher_model,
        embed_direction=args.embed_direction,
        loss_type=args.loss_type,
        callbacks=[RemoveBadSaveCallback(), VerboseEvalCallback()],
    )
    trainer.train_sampler = train_sampler
    trainer.eval_sampler = eval_sampler

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print("🚀  Start training …")
    trainer.train(resume_from_checkpoint=args.continue_from_last_ckpt)
    print("✅  Training finished")
    trainer.args.predict_with_generate = False
    eval_metrics = trainer.evaluate(ds_valid, metric_key_prefix="eval")
    print("🔎  Final evaluation metrics:")
    for k, v in eval_metrics.items():
        print(f"{k:25s}: {v:.4f}")
        
    # if 'sentence1' in ds_test.features and 'sentence2' in ds_test.features:
    #     src_texts = [item['sentence1'] for item in ds_test]
    #     tgt_texts = [item['sentence2'] for item in ds_test]
    #     lang2s = [item['lang2'] for item in ds_test]
    # elif 'anchor' in ds_test.features and 'positive' in ds_test.features:
    #     src_texts = [item['anchor'] for item in ds_test]
    #     tgt_texts = [item['positive'] for item in ds_test]
    #     if 'lang2' in ds_test.features:
    #         lang2s = [item['lang2'] for item in ds_test]
    #     else:
    #         print("Field lang2 tidak ditemukan! Membagi data test berdasarkan pasangan bahasa")
    #         # lang2s = ["jav"] * len(ds_test)
    #         num_per_pair = len(ds_test) // len(args.pairs)
    #         lang2s = []
    #         for pair in args.pairs:
    #             tgt_lang = pair.split("-")[0]  # ambil bahasa low-resource di kiri (sun/jav)
    #             lang2s.extend([tgt_lang] * num_per_pair)
    # else:
    #     raise ValueError("Tidak ditemukan field sentence1/sentence2 atau anchor/positive pada ds_test!")
    # labels = [5.0 for _ in ds_test]  # Atau custom/otomatisasi sesuai evaluasi
    # tokenizer = student_tokenizer  # sudah di-load di pipeline
    # model_pre = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    # scores_pre = get_embedding_scores(model_pre, tokenizer, src_texts, tgt_texts, device)
    # del model_pre; torch.cuda.empty_cache()
    # model_post = model  # sudah di-load, hasil training
    # scores_post = get_embedding_scores(model_post, tokenizer, src_texts, tgt_texts, device)
    # del model_post; torch.cuda.empty_cache()
    #  # --- EVALUASI PER-BAHASA (Pisahkan index Jawa & Sunda)
    # idx_jav = [i for i, l in enumerate(lang2s) if l == "jav"]
    # src_jav = [src_texts[i] for i in idx_jav]
    # tgt_jav = [tgt_texts[i] for i in idx_jav]
    # labels_jav = [labels[i] for i in idx_jav]
    # idx_sun = [i for i, l in enumerate(lang2s) if l == "sun"]
    # src_sun = [src_texts[i] for i in idx_sun]
    # tgt_sun = [tgt_texts[i] for i in idx_sun]
    # labels_sun = [labels[i] for i in idx_sun]
    # print("=== Evaluasi Retrieval/CosSim Indonesia <-> Jawa ===")
    # run_eval(src_jav, tgt_jav, labels_jav, student_tokenizer, model, device="cuda", retrieval=True, error_analysis_k=5)
    # # print("=== Evaluasi Retrieval/CosSim Indonesia <-> Sunda ===")
    # # run_eval(src_sun, tgt_sun, labels_sun, student_tokenizer, model, device="cuda", retrieval=True, error_analysis_k=5)
    # ==== DETEKSI SKEMA DATA TEST ====
    # Ambil daftar fitur/kolom
    features = ds_test.features if hasattr(ds_test, "features") else ds_test.column_names
    
    # Pastikan ada 'lang2' agar evaluasi per-bahasa tidak tercampur
    assert "lang2" in features, "Field 'lang2' wajib ada untuk evaluasi per-bahasa agar tidak tercampur."
    lang2s = [row["lang2"] for row in ds_test]
    
    # Bentuk pasangan sumber–target (dan negatif jika ada)
    if ("sentence1" in features) and ("sentence2" in features):
        # Pairwise (SCL / NT-Xent)
        src_all = [row["sentence1"] for row in ds_test]
        tgt_all = [row["sentence2"] for row in ds_test]
        negatives_all = None
    elif ("anchor" in features) and ("positive" in features):
        # Triplet (atau minimal anchor–positive)
        src_all = [row["anchor"]   for row in ds_test]  # anchors
        tgt_all = [row["positive"] for row in ds_test]  # positives
        negatives_all = [row["negative"] for row in ds_test] if "negative" in features else None
    else:
        raise ValueError("Dataset test harus punya (sentence1,sentence2) atau (anchor,positive).")
    
    # MODE dari loss/skema
    is_triplet_loss = hasattr(args, "loss_type") and (str(args.loss_type).lower() == "triplet")
    mode = "triplet" if (is_triplet_loss or (("anchor" in features) and ("positive" in features))) else "pairwise"
    
    # List bahasa unik (preserve order)
    langs_order, _seen = [], set()
    for l in lang2s:
        if l not in _seen:
            langs_order.append(l)
            _seen.add(l)
    
    # Evaluasi per-bahasa
    for lang_code in langs_order:
        idx = [i for i, l in enumerate(lang2s) if l == lang_code]
        if not idx:
            continue
    
        print("\n" + ("=" * 10) + f" Evaluasi ({mode}) lang2={lang_code} " + ("=" * 10))
    
        if mode == "triplet":
            anchors   = [src_all[i] for i in idx]
            positives = [tgt_all[i] for i in idx]
            negatives = (
                [negatives_all[i] for i in idx]
                if (negatives_all is not None and len(negatives_all) == len(lang2s))
                else None
            )
    
            res = run_eval(
                src_texts=anchors,
                tgt_texts=positives,
                labels=None,
                tokenizer=student_tokenizer,
                model=model,            # model student hasil training
                device="cuda",
                mode="triplet",
                negatives=negatives,    # boleh None
                ks=(1,),
                max_length=128,
                layer=-1,
                batch_size=32,
                verbose=True,
                retrieval=True,
            )
    
        else:  # pairwise
            src_lang = [src_all[i] for i in idx]
            tgt_lang = [tgt_all[i] for i in idx]
    
            res = run_eval(
                src_texts=src_lang,
                tgt_texts=tgt_lang,
                labels=None,
                tokenizer=student_tokenizer,
                model=model,
                device="cuda",
                mode="pairwise",
                ks=(1,),
                max_length=128,
                layer=-1,
                batch_size=32,
                verbose=True,
                retrieval=True,
            )
    
        # Ringkas output
        for k, v in res.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Guard kecil: pastikan ada pad_token (penting untuk model decoder-only)
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = getattr(student_tokenizer, "eos_token", None) or student_tokenizer.unk_token
    
    # Tentukan pasangan bahasa FLORES yang ingin diuji
    # Gunakan kode NLLB/FLORES (mis. ind_Latn, jav_Latn, sun_Latn)
    # pairs = []
    # if any(l in langs_order for l in ("jav", "jv")):
    #     pairs.append(("ind_Latn", "jav_Latn"))
    # if any(l in langs_order for l in ("sun", "su")):
    #     pairs.append(("ind_Latn", "sun_Latn"))

    # pairs = [("ind_Latn", "jav_Latn")]     
    pairs = [("ind_Latn", "sun_Latn")]     
    # pairs = [("eng_Latn", "jav_Latn")]      
    # pairs = [("eng_Latn", "sun_Latn")] 
    # eval_langs = sorted({l for p in pairs for l in p})  # -> ["ind_Latn", "jav_Latn"]
    
    if pairs:
        print("\n========== Evaluasi MTEB FloresBitextMining ==========")
        mteb_res = eval_mteb.run_mteb_flores(
            model=model,
            tokenizer=student_tokenizer,
            pairs=pairs,          # contoh: [("ind_Latn","jav_Latn")]
            device="cuda",
            max_length=128,
            batch_size=16,
            pooling="mean",           # konsisten dengan training/eval internal
            eval_splits=("devtest",), # FLORES tidak punya 'test'
            # output_folder="mteb_results/imascl_flores",
        )
        if not mteb_res:
            print("[WARN] MTEB result kosong. Cek [DBG] raw evaluator keys di atas.")
        for k, v in sorted(mteb_res.items()):
            print(f"mteb.{k}: {v}")

if __name__ == "__main__":
    main()
