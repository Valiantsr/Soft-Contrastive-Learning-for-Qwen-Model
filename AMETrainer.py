from transformers import Trainer
import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Optional
from torch.utils.data import DataLoader
from modules.AMESampler import LengthInBatchSampler

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07):
    assert z1.dim() == 2 and z2.dim() == 2, "z1/z2 harus [B, D]"
    assert z1.shape == z2.shape, "z1 dan z2 harus punya shape yang sama"
    B, D = z1.shape
    device = z1.device
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)               # [2B, D]
    sim = torch.mm(z, z.t()) / tau               # [2B, 2B]
    mask = torch.eye(2*B, device=device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float('-inf'))
    pos = torch.arange(B, device=device)
    labels = torch.cat([pos + B, pos], dim=0)    # [2B]
    loss = F.cross_entropy(sim, labels)
    # --- sanity check ringan (opsional, non-blocking) ---
    with torch.no_grad():
        pos_sim = (z1 * z2).sum(dim=-1)          # cosine untuk pasangan benar
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("NT-Xent loss NaN/Inf: cek input atau tau.")
        # contoh: print ringkas (matikan di produksi)
        # print(f"[NTX] B={B} tau={tau} | pos_sim mean={pos_sim.mean():.3f}")
    return loss

def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    loss = F.relu(margin + neg_sim - pos_sim).mean()
    return loss

class AMETrainer(Trainer):
    def __init__(
        self,
        *args,
        scloss_fn=None,
        teacher_model=None,
        loss_type=None,
        embed_direction="both",
        pooling_fn=None,
        dropout_prob=0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.embed_direction = embed_direction
        self.loss_type = loss_type
        self.scloss_fn = scloss_fn or self._default_scloss
        self.pooling_fn = pooling_fn if pooling_fn is not None else self.default_pooling_fn
        self.embed_model_dir = getattr(self.args, "embed_model_dir", None)
        self.dropout = nn.Dropout(dropout_prob)
        if self.embed_model_dir and self.embed_model_dir != "self":
            from transformers import AutoModel
            self.alt_teacher = AutoModel.from_pretrained(
                self.embed_model_dir,
                output_hidden_states=True
            ).to(self.model.device)
            self.alt_teacher.eval()
        else:
            self.alt_teacher = None

    def _default_scloss(
        self,
        embeds_1,                 # Student: src
        embeds_2,                 # Student: tgt
        teacher_embeds_1=None,    # Teacher: src
        teacher_embeds_2=None,    # Teacher: tgt
        teacher_matrix=None,
        tau=None,
        margin=0.1,
        w_func="softmax",
        reduction="mean",):      # Soft-label matrix (optional, override):
        # Student similarities
        if tau is None:
            tau = self.args.tau
        e1 = torch.nn.functional.normalize(embeds_1, dim=-1)
        e2 = torch.nn.functional.normalize(embeds_2, dim=-1)
        B = e1.size(0)
        hard_labels = torch.arange(B, device=e1.device)
        logits = torch.mm(e1, e2.t()) / tau
        # --- Teacher/Soft Label Matrix ---
        if teacher_matrix is not None:
            # Sudah diberikan dari luar (priority/average)
            sim_t = teacher_matrix / tau
            mask = torch.eye(B, device=sim_t.device, dtype=torch.bool)
            sim_t = sim_t.masked_fill(mask, -1e9)
            soft_lbl = torch.softmax(sim_t, dim=-1)
            # soft_lbl = teacher_matrix
        elif teacher_embeds_1 is not None and teacher_embeds_2 is not None:
            t1 = torch.nn.functional.normalize(teacher_embeds_1, dim=-1)
            t2 = torch.nn.functional.normalize(teacher_embeds_2, dim=-1)
            sim_t = torch.mm(t1, t2.t()) / tau
            mask = torch.eye(B, device=sim_t.device).bool()
            sim_t = sim_t.masked_fill(mask, -1e9)
            if w_func.lower() == "softmax":
                soft_lbl = torch.softmax(sim_t, dim=-1)
            else:
                soft_lbl = (sim_t > margin).float()
        else:
            soft_lbl = None 
        # --- Row/Column Loss: (cross + distill, atau hanya hard) ---
        # Student-Student (N x N) untuk cross-entropy (row-wise)
        ce_row = torch.nn.functional.cross_entropy(logits, hard_labels, reduction=reduction)
        if soft_lbl is not None:
            epsilon = 1e-8
            soft_lbl = soft_lbl + epsilon
            soft_lbl = soft_lbl / soft_lbl.sum(dim=1, keepdim=True)
            # hitung KL‐div row‐wise
            kl_row = torch.nn.functional.kl_div(
                torch.log_softmax(logits, dim=-1),
                soft_lbl,
                reduction="batchmean")
            # gabungkan CE + KL
            total_row = (ce_row + kl_row) / 2
        else:
            total_row = ce_row
        # Optional: column-wise (symmetry, seperti di paper SCL aslinya)
        ce_col = torch.nn.functional.cross_entropy(logits.t(), hard_labels, reduction=reduction)
        if soft_lbl is not None:
            eps = 1e-8
            soft_lbl_col = soft_lbl.t()
            soft_lbl_col = (soft_lbl_col + eps) / soft_lbl_col.sum(dim=1, keepdim=True)
            kl_col = torch.nn.functional.kl_div(
                torch.log_softmax(logits.t(), dim=-1),
                soft_lbl_col,
                reduction="batchmean"
            )
            total_col = (ce_col + kl_col) / 2
        else:
            total_col = ce_col
        # Combine row & col
        loss = (total_row + total_col) / 2
        if soft_lbl is not None:
            row_sums = soft_lbl.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                "soft_lbl rows not normalized (cek /tau, masking, softmax)"
            assert soft_lbl.diag().max().item() < 1e-6, \
                "soft_lbl diagonal harus ≈0 (mask diag sebelum softmax)"
            # print(f"[DBG] tau={tau}, B={soft_lbl.size(0)}, "
            #       f"soft_lbl mean={soft_lbl.mean().item():.4f}, "
            #       f"max={soft_lbl.max().item():.4f}, min={soft_lbl.min().item():.4f}")
        else:
            print("[DBG] soft_lbl is None → hanya CE (tanpa KL).")
        return loss
    
    def default_pooling_fn(self, hidden_states, attention_mask):
        last_hidden = hidden_states[-1] if isinstance(hidden_states, (list, tuple)) else hidden_states
        mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_exp, dim=1)
        sum_mask = mask_exp.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def compute_similarity(x, y):
        return torch.matmul(
            torch.nn.functional.normalize(x, dim=-1),
            torch.nn.functional.normalize(y, dim=-1).T)

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: int | None = None,):
        dev = next(model.parameters()).device
        for k, v in inputs.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        inputs[k][kk] = vv.to(dev)
            elif isinstance(v, torch.Tensor):
                inputs[k] = v.to(dev)
        if "anchor_input_ids" in inputs:
        # Triplet batch mode
            anchor_ids = inputs["anchor_input_ids"]
            anchor_mask = inputs["anchor_attention_mask"]
            positive_ids = inputs["positive_input_ids"]
            positive_mask = inputs["positive_attention_mask"]
            negative_ids = inputs["negative_input_ids"]
            negative_mask = inputs["negative_attention_mask"]
    
            # Encode anchor/positive/negative
            out_anchor = model(input_ids=anchor_ids, attention_mask=anchor_mask, output_hidden_states=True, return_dict=True)
            out_positive = model(input_ids=positive_ids, attention_mask=positive_mask, output_hidden_states=True, return_dict=True)
            out_negative = model(input_ids=negative_ids, attention_mask=negative_mask, output_hidden_states=True, return_dict=True)
            # Pooling
            anchor_emb = self.pooling_fn(out_anchor.hidden_states[-1], anchor_mask)
            positive_emb = self.pooling_fn(out_positive.hidden_states[-1], positive_mask)
            negative_emb = self.pooling_fn(out_negative.hidden_states[-1], negative_mask)
            # Triplet loss (gunakan triplet_loss fn)
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb, margin=0.2)
            if return_outputs:
                # Output logits bisa diatur sesuai kebutuhan (misal: stack 3 embeddings)
                out = {"anchor_emb": anchor_emb, "positive_emb": positive_emb, "negative_emb": negative_emb}
                return loss, out
            return loss
            
        ids_main = inputs["batch_main"]["input_ids"]
        msk_main = inputs["batch_main"]["attention_mask"]
        out = model(
            input_ids=ids_main,
            attention_mask=msk_main,
            output_hidden_states=True,
            return_dict=True,
        )
        stud_emb = self.pooling_fn(out.hidden_states[-1], msk_main)
        self.dropout.train(model.training)  
        stud_emb = self.dropout(stud_emb)
        B = stud_emb.size(0) // 2
        stud1, stud2 = stud_emb[:B], stud_emb[B:]
    
        teach1 = teach2 = None
        if self.teacher_model is not None:
            with torch.no_grad():
                if self.embed_direction in ("left", "both", "input"):
                    ids1 = inputs["batch_1"]["input_ids"]
                    msk1 = inputs["batch_1"]["attention_mask"]
                    tout1 = self.teacher_model(
                        input_ids=ids1,
                        attention_mask=msk1,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    teach1 = self.pooling_fn(tout1.hidden_states[-1], msk1)
                    assert teach1 is not None, "teach1 harusnya bukan None"
                if self.embed_direction in ("right", "both"):
                    ids2 = inputs["batch_2"]["input_ids"]
                    msk2 = inputs["batch_2"]["attention_mask"]
                    tout2 = self.teacher_model(
                        input_ids=ids2,
                        attention_mask=msk2,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    teach2 = self.pooling_fn(tout2.hidden_states[-1], msk2)
                    assert teach2 is not None, "teach2 harusnya bukan None"
        # assert teach1 is not None and teach2 is not None, "Teacher embeddings belum di‐populate"
        if self.embed_direction in ("left", "input"):
            assert teach1 is not None, "Teacher embeddings (left/input) belum di-populate"
        if self.embed_direction == "right":
            assert teach2 is not None, "Teacher embeddings (right) belum di-populate"
        if self.embed_direction == "both":
            assert teach1 is not None and teach2 is not None, "Teacher embeddings (both) belum di-populate"


        if hasattr(self, "alt_teacher") and self.alt_teacher is not None:
            with torch.no_grad():
                if self.embed_direction in ("left", "both", "input"):
                    out_alt1 = self.alt_teacher(
                        input_ids=inputs["batch_1"]["input_ids"],
                        attention_mask=inputs["batch_1"]["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    teach1 = self.pooling_fn(out_alt1.hidden_states[-1],
                                             inputs["batch_1"]["attention_mask"])
                if self.embed_direction in ("right", "both"):
                    out_alt2 = self.alt_teacher(
                        input_ids=inputs["batch_2"]["input_ids"],
                        attention_mask=inputs["batch_2"]["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    teach2 = self.pooling_fn(out_alt2.hidden_states[-1],
                                             inputs["batch_2"]["attention_mask"])

        loss_type = getattr(self, "loss_type", "scl")
        if loss_type == "scl":
            label_mode = getattr(self, "label_mode", "average")
            if label_mode == "average":
                if self.embed_direction != "both":
                    raise ValueError("label_mode='average' hanya untuk embed_direction='both'")
                sim1 = self.compute_similarity(teach1, teach1)
                sim2 = self.compute_similarity(teach2, teach2)
                M = 0.5 * (sim1 + sim2)
                # print(f"[DBG] SCL path=matrix | embed_direction={self.embed_direction} | label_mode={label_mode}")
                loss = self.scloss_fn(embeds_1=stud1, embeds_2=stud2, teacher_matrix=M, tau=self.args.tau, w_func=self.args.w_func)
            elif label_mode == "priority":
                # print(f"[DBG] SCL path=embeds | embed_direction={self.embed_direction} | label_mode={label_mode}")
                if self.embed_direction == "both":
                    loss = self.scloss_fn(stud1, stud2, teach1, teach2, tau=self.args.tau, w_func=self.args.w_func)
                elif self.embed_direction == "left":
                    # duplikasi t1 agar KL tetap hidup
                    loss = self.scloss_fn(stud1, stud2, teach1, teach1, tau=self.args.tau, w_func=self.args.w_func)
                elif self.embed_direction == "right":
                    loss = self.scloss_fn(stud1, stud2, teach2, teach2, tau=self.args.tau, w_func=self.args.w_func)
                elif self.embed_direction == "input":
                    # kalau “input” maksudnya anchor=left, samakan dengan left
                    loss = self.scloss_fn(stud1, stud2, teach1, teach1, tau=self.args.tau, w_func=self.args.w_func)
                else:
                    raise ValueError("Unknown embed_direction")
            else:
                raise ValueError("Unknown label_mode")
        elif loss_type == "nt_xent":
            loss = nt_xent_loss(stud1, stud2, tau=0.07)
        elif loss_type == "triplet":
            idx = torch.randperm(B)
            negative = stud2[idx]
            loss = triplet_loss(stud1, stud2, negative, margin=0.2)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if hasattr(self, "_total_loss") and (num_items_in_batch is not None):
            self._total_loss += loss.detach() * num_items_in_batch
            self._total_loss_samples += num_items_in_batch

        out.logits = torch.stack([stud1, stud2], dim=1)
        return (loss, out) if return_outputs else loss

    def training_step(self, *args, **kwargs):
        # 1) Gunakan Trainer default untuk forward, backward, dan optimizer step
        loss = super().training_step(*args, **kwargs)
        # 2) Verifikasi: teacher_model tidak boleh punya grad
        model, inputs = args[0], args[1]
        if self.teacher_model is not None:
            for name, p in self.teacher_model.named_parameters():
                if p.grad is not None:
                    raise RuntimeError(
                        f"⚠️ Teacher parameter `{name}` menerima grad!")
        # 3) Verifikasi: semua parameter student yang requires_grad pasti punya grad
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                 raise RuntimeError(
                    f"❌ Student parameter `{name}` tidak punya grad!")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss, outputs = self.compute_loss(
                model, inputs, return_outputs=True)
            # ========== Handling TRIPLET ==========
            if isinstance(outputs, dict) and "anchor_emb" in outputs:
                anchor_emb = outputs["anchor_emb"]
                positive_emb = outputs["positive_emb"]
                negative_emb = outputs["negative_emb"]
                # Gabungkan atau pilih salah satu untuk alignment evaluasi
                # Di sini, misal hanya anchor dan positive yang dipakai (umum untuk retrieval/align)
                alignment_out = torch.stack([anchor_emb, positive_emb], dim=1)  # [B,2,D]
                B = anchor_emb.size(0)
        # ========== Handling SCL / NT_XENT ==========
            elif hasattr(outputs, "hidden_states"):
                last_hidden = outputs.hidden_states[-1]
                attention_mask = inputs["batch_main"]["attention_mask"]
                embeddings = self.pooling_fn(last_hidden, attention_mask)
                B = embeddings.size(0) // 2
                stud1, stud2 = embeddings[:B], embeddings[B:]
                alignment_out = torch.stack([stud1, stud2], dim=1)  # [B,2,D]
            else:
                raise ValueError("Unknown output structure in prediction_step.")
        # ==== DEFAULT (ALIGNMENT) ====
        dummy_labels = torch.zeros(B, dtype=torch.long, device=alignment_out.device)
        return (loss, alignment_out, dummy_labels)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.train_sampler,    # pakai sampler yang sudah Anda simpan di self
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,)

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=self.eval_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
