import torch
from transformers import TrainerCallback
import os
import shutil
import sacrebleu
import numpy as np

class AMEMetric:
    def __init__(self):
        pass

    def __call__(self, pred):
        results = {}
        arr = pred.predictions
        if isinstance(arr, tuple):
            arr = arr[0]
        # --- alignment metrics ---
        with torch.no_grad():
            # jika bentuknya [B, 2*D], ubah jadi [B,2,D]
            if arr.ndim == 2:
                B, twoD = arr.shape
                D = twoD // 2
                arr = arr.reshape(B, 2, D)
            embeds = torch.tensor(arr, dtype=torch.float32)
            z1, z2 = embeds[:,0], embeds[:,1]
            z1 = torch.nn.functional.normalize(z1, dim=-1)
            z2 = torch.nn.functional.normalize(z2, dim=-1)
            sim = z1 @ z2.T            # [B,B]
            idx_1_to_2 = sim.argmax(1)
            idx_2_to_1 = sim.argmax(0)
            B = sim.size(0)
            acc_1_to_2 = (idx_1_to_2 == torch.arange(B)).float().mean().item()
            acc_2_to_1 = (idx_2_to_1 == torch.arange(B)).float().mean().item()
            results.update({
                "eval_alignment_acc_1_to_2": acc_1_to_2,
                "eval_alignment_acc_2_to_1": acc_2_to_1,
                "eval_alignment_acc_avg": (acc_1_to_2+acc_2_to_1)/2,
            })
        return results

class RemoveBadSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        best_model_steps = {}
        best_model_metric = {}
        metric_for_save_model_langs = []
        larger_better_dict = {}

        for logs in state.log_history:
            for idx, metric_portion in enumerate(args.metrics_for_save_model):
                for key in logs.keys():
                    if (metric_portion in key) and ("per_second" not in key) and key not in metric_for_save_model_langs:
                        metric_for_save_model_langs.append(key)
                        best_model_steps[key] = -1
                        best_model_metric[key] = 0 if args.metrics_larger_better[idx] else float("inf")
                        larger_better_dict[key] = args.metrics_larger_better[idx]
            
            for metric in metric_for_save_model_langs:
                if metric in logs:
                    if larger_better_dict[metric]:
                        if logs[metric] >= best_model_metric[metric]:
                            best_model_metric[metric] = logs[metric]
                            best_model_steps[metric] = logs["step"]
                    else:
                        if logs[metric] <= best_model_metric[metric]:
                            best_model_metric[metric] = logs[metric]
                            best_model_steps[metric] = logs["step"]

        print("[INFO] Best model steps per metric:", best_model_steps)

        for dirname in os.listdir(args.output_dir):
            if dirname.startswith("checkpoint-"):
                step = int(dirname.split("-")[-1])
                if step not in best_model_steps.values():
                    checkpoint_path = os.path.join(args.output_dir, dirname)
                    print(f"[INFO] Removing non-best checkpoint: {checkpoint_path}")
                    shutil.rmtree(checkpoint_path, ignore_errors=True)

class VerboseEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        print("\n=== evaluation ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k:25s}: {v:.4f}")
            else:
                print(f"{k:25s}: {v}")
        print("==================\n")
