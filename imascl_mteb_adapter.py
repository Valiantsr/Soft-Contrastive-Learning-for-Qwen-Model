import torch, numpy as np
from typing import List

class IMASCLAdapter:
    def __init__(self, model, tokenizer, device=None, max_length=128, batch_size=32, pooling="mean"):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        if self.tok.pad_token_id is None:                 # guard penting utk decoder-only
            if getattr(self.tok, "eos_token", None) is not None:
                self.tok.pad_token = self.tok.eos_token
            elif getattr(self.tok, "unk_token", None) is not None:
                self.tok.pad_token = self.tok.unk_token

    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        embs = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            inputs = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                              max_length=self.max_length).to(self.device)
            out = self.model(**inputs, return_dict=True, output_hidden_states=True)
            rep = out.hidden_states[-1]                                  # (B,T,H)
            mask = inputs["attention_mask"].unsqueeze(-1).to(rep.dtype)  # (B,T,1)

            x = (rep * mask).sum(1) / mask.sum(1).clamp(min=1e-6)        # mean pooling
            if self.pooling == "cls":
                x = rep[:, 0]
            elif self.pooling in ("eos", "last"):
                idx = inputs["attention_mask"].sum(1) - 1
                x = rep[torch.arange(rep.size(0), device=rep.device), idx]

            x = torch.nn.functional.normalize(x, p=2, dim=-1)            # L2 norm
            embs.append(x.detach().cpu().numpy())
        return np.concatenate(embs, axis=0)
