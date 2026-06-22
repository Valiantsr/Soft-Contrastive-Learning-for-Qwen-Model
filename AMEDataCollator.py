from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers.data.data_collator import DefaultDataCollator
from transformers import logging

logger = logging.get_logger(__name__)

# Tokenizer student (Qwen)
from transformers import AutoTokenizer
student_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
student_tokenizer.padding_side = "left"
student_tokenizer.truncation_side = "right"
student_tokenizer.model_max_length = 512
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token

# Tokenizer teacher (QWEN)
teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

def pad_tensor(tensor_list, pad_val, padding_side, multiple=None):
    if padding_side == "right":
        padded = pad_sequence(tensor_list, batch_first=True, padding_value=pad_val)
    else:
        rev = [torch.flip(t, dims=[0]) for t in tensor_list]
        padded = pad_sequence(rev, batch_first=True, padding_value=pad_val)
        padded = torch.flip(padded, dims=[1])
    if multiple is not None:
        L = padded.size(1)
        if L % multiple != 0:
            new_L = ((L // multiple) + 1) * multiple
            delta = new_L - L
            extra = padded.new_full((padded.size(0), delta), pad_val)
            padded = torch.cat([extra, padded], dim=1) if padding_side == "left" else torch.cat([padded, extra], dim=1)
    return padded

def clamp_oov(tensor, vocab_size, pad_id):
    tensor = tensor.clone()
    tensor[tensor >= vocab_size] = pad_id
    return tensor

class AMEDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, pad_to_multiple_of=None, device=None, embed_direction="both"):
        self.tok = tokenizer
        self.multiple = pad_to_multiple_of
        self.device = device
        self.embed_direction = embed_direction

    def __call__(self, features: List[Dict[str, Any]], return_tensors: str | None = None):
        pad_id = self.tok.pad_token_id
        if pad_id is None or pad_id >= self.tok.vocab_size:
            pad_id = self.tok.eos_token_id
        if pad_id is None or pad_id >= self.tok.vocab_size:
            pad_id = self.tok.vocab_size - 1

        def collect(key):
            return [torch.tensor(f[key], dtype=torch.long) for f in features]

        # ===== PATCH: HANDLE TRIPLET DATASET =====
        if (
            "anchor_input_ids" in features[0]
            and "positive_input_ids" in features[0]
            and "negative_input_ids" in features[0]
        ):
            anchor_ids   = collect("anchor_input_ids")
            anchor_mask  = collect("anchor_attention_mask")
            positive_ids = collect("positive_input_ids")
            positive_mask= collect("positive_attention_mask")
            negative_ids = collect("negative_input_ids")
            negative_mask= collect("negative_attention_mask")

            anchor_ids   = pad_tensor(anchor_ids, pad_id, self.tok.padding_side, self.multiple)
            anchor_mask  = pad_tensor(anchor_mask, 0, self.tok.padding_side, self.multiple)
            positive_ids = pad_tensor(positive_ids, pad_id, self.tok.padding_side, self.multiple)
            positive_mask= pad_tensor(positive_mask, 0, self.tok.padding_side, self.multiple)
            negative_ids = pad_tensor(negative_ids, pad_id, self.tok.padding_side, self.multiple)
            negative_mask= pad_tensor(negative_mask, 0, self.tok.padding_side, self.multiple)

            batch_triplet = {
                "anchor_input_ids": anchor_ids,
                "anchor_attention_mask": anchor_mask,
                "positive_input_ids": positive_ids,
                "positive_attention_mask": positive_mask,
                "negative_input_ids": negative_ids,
                "negative_attention_mask": negative_mask,
            }
            # Clamp OOV
            for k, v in batch_triplet.items():
                if torch.is_tensor(v):
                    batch_triplet[k] = clamp_oov(v, self.tok.vocab_size, pad_id)
            if self.device is not None:
                for k, v in batch_triplet.items():
                    if torch.is_tensor(v):
                        batch_triplet[k] = v.to(self.device, non_blocking=True)
            return batch_triplet

        # ===== END PATCH TRIPLET =====

        ids1 = collect("input_ids_1")
        msk1 = collect("attention_mask_1")
        ids2 = collect("input_ids_2")
        msk2 = collect("attention_mask_2")

        ids1 = pad_tensor(ids1, pad_id, self.tok.padding_side, self.multiple)
        msk1 = pad_tensor(msk1, 0, self.tok.padding_side, self.multiple)
        ids2 = pad_tensor(ids2, pad_id, self.tok.padding_side, self.multiple)
        msk2 = pad_tensor(msk2, 0, self.tok.padding_side, self.multiple)

        L1, L2 = ids1.size(1), ids2.size(1)
        if L1 != L2:
            L_max = max(L1, L2)
            def pad_to(tensor, pad_value):
                diff = L_max - tensor.size(1)
                if diff == 0:
                    return tensor
                pad = tensor.new_full((tensor.size(0), diff), pad_value)
                return torch.cat([pad, tensor], dim=1) if self.tok.padding_side == "left" else torch.cat([tensor, pad], dim=1)

            ids1, ids2 = pad_to(ids1, pad_id), pad_to(ids2, pad_id)
            msk1, msk2 = pad_to(msk1, 0), pad_to(msk2, 0)

        ids_main = torch.cat([ids1, ids2], dim=0)
        msk_main = torch.cat([msk1, msk2], dim=0)

        batch = {
            "batch_main": {"input_ids": ids_main, "attention_mask": msk_main},
            "batch_1": {"input_ids": ids1, "attention_mask": msk1},
            "batch_2": {"input_ids": ids2, "attention_mask": msk2},
        }

        if "embed_input_ids_1" in features[0]:
            teacher_pad_id = teacher_tokenizer.pad_token_id
            if teacher_pad_id is None or teacher_pad_id >= teacher_tokenizer.vocab_size:
                teacher_pad_id = teacher_tokenizer.eos_token_id
            if teacher_pad_id is None or teacher_pad_id >= teacher_tokenizer.vocab_size:
                teacher_pad_id = teacher_tokenizer.vocab_size - 1
            eids1 = collect("embed_input_ids_1")
            emsk1 = collect("embed_attention_mask_1")
            eids1 = pad_tensor(eids1, teacher_tokenizer.pad_token_id, teacher_tokenizer.padding_side, self.multiple)
            emsk1 = pad_tensor(emsk1, 0, teacher_tokenizer.padding_side, self.multiple)
            batch["embed_batch_1"] = {"input_ids": eids1, "attention_mask": emsk1}

        if "embed_input_ids_2" in features[0]:
            teacher_pad_id = teacher_tokenizer.pad_token_id
            if teacher_pad_id is None or teacher_pad_id >= teacher_tokenizer.vocab_size:
                teacher_pad_id = teacher_tokenizer.eos_token_id
            if teacher_pad_id is None or teacher_pad_id >= teacher_tokenizer.vocab_size:
                teacher_pad_id = teacher_tokenizer.vocab_size - 1
            eids2 = collect("embed_input_ids_2")
            emsk2 = collect("embed_attention_mask_2")
            eids2 = pad_tensor(eids2, teacher_tokenizer.pad_token_id, teacher_tokenizer.padding_side, self.multiple)
            emsk2 = pad_tensor(emsk2, 0, teacher_tokenizer.padding_side, self.multiple)
            batch["embed_batch_2"] = {"input_ids": eids2, "attention_mask": emsk2}

        for bname in ["batch_1", "batch_2"]:
            max_id = batch[bname]["input_ids"].max().item()
            vocab_size = self.tok.vocab_size
            # if max_id > vocab_size - 1:
            #     logger.warning(f"[⚠️ ID OVERRUN] {bname} input_ids max={max_id} exceeds vocab_size={vocab_size}")
            #     print(f"[⚠️ ID OVERRUN] {bname} input_ids max={max_id} exceeds vocab_size={vocab_size}")

        # batch["labels"] = ids2
        batch["labels"] = clamp_oov(ids2, self.tok.vocab_size, pad_id)
        # if self.device is not None:
        #     for k, v in batch.items():
        #         for kk in v:
                    # batch[k][kk] = v[kk].to(self.device, non_blocking=True)
        for k, v in batch.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        batch[k][kk] = clamp_oov(vv, self.tok.vocab_size, pad_id)
            elif torch.is_tensor(v):
                batch[k] = clamp_oov(v, self.tok.vocab_size, pad_id)

        for k, v in batch.items():
            if isinstance(v, dict) and "input_ids" in v:
                over = (v["input_ids"] >= self.tok.vocab_size).sum().item()
                if over > 0:
                    print(f"[WARNING] OOV token di {k}: {over} dari {v['input_ids'].numel()}")
            if k == "labels" and torch.is_tensor(v):
                over = (v >= self.tok.vocab_size).sum().item()
                if over > 0:
                    print(f"[WARNING] OOV token di labels: {over} dari {v.numel()}")

        return batch
