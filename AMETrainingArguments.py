# ─── AMETrainingArguments.py (ganti seluruh isi) ─────────────────────────
from dataclasses import dataclass, field
from transformers import TrainingArguments, IntervalStrategy
from transformers.utils import add_start_docstrings
from transformers.generation import GenerationConfig

@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class AMETrainingArguments(TrainingArguments):
    # ────── field ekstra untuk SCL/AME ────────────────────────────────
    cross_acc_sampling : bool  = field(default=False)
    dataset_pseudo     : str   = field(default="None")
    margin             : float = field(default=0.3)
    positive_threshold : float = field(default=-1.0)
    loss_type          : str   = field(default=None)
    label_mode         : str   = field(default=None)
    tau                : float = field(default=0.1)
    w_func             : str   = field(default="raw")
    gpu_num            : int   = field(default=1)
    embed_model_dir    : str   = field(default="self")
    embed_direction    : str   = field(default="both")
    train_mono_space   : bool  = field(default=False)
    small_batch_size   : int   = field(default=None)
    predict_with_generate   : bool  = field(default=False)
    generation_max_length   : int   = field(default=128)
    generation_max_new_tokens : int = field(default=128)
    max_new_tokens          : int   = field(default=None)
    metrics_for_save_model  : tuple = field(default=("eval_loss",))
    metrics_larger_better   : tuple = field(default=(False,))
    dataset_sentence_idx_dict : dict= field(default=None)

    # ────── tangkap semua argumen HF lain lewat **kwargs ──────────────
    def __init__(self, predict_with_generate=False, generation_max_length=128, generation_max_new_tokens: int = 128, **kwargs):
        # ambil field khusus kita lalu buang dari hf_args
        self.margin              = kwargs.pop("margin", 0.3)
        self.positive_threshold  = kwargs.pop("positive_threshold", -1.0)
        self.loss_type           = kwargs.pop("loss_type", None)
        self.label_mode          = kwargs.pop("label_mode", None)
        self.tau                 = kwargs.pop("tau", 0.1)
        self.w_func              = kwargs.pop("w_func", "raw")
        self.cross_acc_sampling  = kwargs.pop("cross_acc_sampling", False)
        self.dataset_pseudo      = kwargs.pop("dataset_pseudo", "None")
        self.gpu_num             = kwargs.pop("gpu_num", 1)
        self.embed_model_dir     = kwargs.pop("embed_model_dir", "self")
        self.embed_direction     = kwargs.pop("embed_direction", "both")
        self.train_mono_space    = kwargs.pop("train_mono_space", False)
        self.small_batch_size    = kwargs.pop("small_batch_size", None)
        self.predict_with_generate = kwargs.pop("predict_with_generate", predict_with_generate)
        self.generation_max_length = kwargs.pop("generation_max_length", generation_max_length)
        self.generation_max_new_tokens = kwargs.pop("generation_max_new_tokens", generation_max_new_tokens)
        self.metrics_for_save_model = kwargs.pop(
            "metrics_for_save_model", ("eval_loss",)
        )
        self.metrics_larger_better  = kwargs.pop(
            "metrics_larger_better", (False,)
        )
        self.dataset_sentence_idx_dict = kwargs.pop(
            "dataset_sentence_idx_dict", None
        )
        eval_strat   = kwargs.pop("evaluation_strategy", IntervalStrategy.STEPS)
        save_strat   = kwargs.pop("save_strategy",       IntervalStrategy.STEPS)
        log_strat    = kwargs.pop("logging_strategy",    IntervalStrategy.STEPS)
        best_at_end  = kwargs.pop("load_best_model_at_end", False)

        super().__init__(**kwargs)
        self.evaluation_strategy    = eval_strat
        self.save_strategy          = save_strat
        self.logging_strategy       = log_strat
        self.load_best_model_at_end = best_at_end
        self.predict_with_generate  = predict_with_generate
        self.generation_max_length  = generation_max_length
        self.generation_max_new_tokens = generation_max_new_tokens

    # opsional: custom to_dict agar GenerationConfig aman‐serialise
    def to_dict(self):
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d
