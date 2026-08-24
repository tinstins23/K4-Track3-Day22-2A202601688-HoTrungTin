#!/usr/bin/env python3
"""CLI wrapper for NB3 logic — trains a DPO adapter.

Usage:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --beta 0.05 --output-dir adapters/dpo-b0.05
    python scripts/train_dpo.py --beta 0.5  --output-dir adapters/dpo-b0.50

Mirrors `notebooks/03_dpo_train.py`. Used by `make beta-sweep` for the rigor
add-on +6 (β-sweep mini-experiment).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sft-path", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--pref-path", default=str(REPO / "data" / "pref" / "train.parquet"))
    parser.add_argument("--output-dir", default=str(REPO / "adapters" / "dpo"))
    args = parser.parse_args()

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    if tier == "T4":
        base_model = "unsloth/Qwen2.5-3B-bnb-4bit"
        max_len, max_prompt = 256, 128
        batch, grad_accum = 1, 16
    else:
        base_model = "unsloth/Qwen2.5-7B-bnb-4bit"
        max_len, max_prompt = 1024, 512
        batch, grad_accum = 1, 4

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Tier:       {tier}")
    print(f"Base:       {base_model}")
    print(f"Beta / LR:  {args.beta} / {args.lr}")
    print(f"Output:     {output}")

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=max_len, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, args.sft_path, is_trainable=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42, use_rslora=False, loftq_config=None,
    )

    config = DPOConfig(
        output_dir=str(output.parent / f"{output.name}-checkpoints"),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=max_len,
        max_prompt_length=max_prompt,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=42,
        loss_type="sigmoid",
        report_to="none",
        gradient_checkpointing=False,
        precompute_ref_log_probs=True,
    )

    pref = Dataset.from_parquet(args.pref_path)
    trainer = DPOTrainer(
        model=model, ref_model=None, args=config,
        train_dataset=pref, processing_class=tokenizer,
    )

    # T4: keep xformers, flatten GQA 5D BMGHK → 4D (SDPA OOMs on 14GB).
    if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
        import importlib

        try:
            import xformers  # noqa: F401
            has_xf = True
        except Exception:
            has_xf = False

        def _backend(use_varlen=False):
            try:
                from unsloth.models._utils import HAS_FLASH_ATTENTION as _fa
            except Exception:
                _fa = False
            if _fa:
                return "flash_varlen" if use_varlen else "flash_dense"
            return "xformers" if has_xf else "sdpa"

        for name in (
            "unsloth.utils.attention_dispatch",
            "unsloth.models._utils",
            "unsloth.models.llama",
            "unsloth.models.qwen2",
        ):
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            if hasattr(mod, "HAS_XFORMERS"):
                mod.HAS_XFORMERS = has_xf
            if hasattr(mod, "select_attention_backend"):
                mod.select_attention_backend = _backend

        def _wrap(orig):
            inner = orig
            while getattr(inner, "_lab22_orig", None) is not None:
                inner = inner._lab22_orig
            if getattr(inner, "_lab22_t4", False):
                return inner

            def wrapped(query, key, value, attn_bias=None, p=0.0, *args, **kwargs):
                q, k, v, restore = query, key, value, None
                if getattr(q, "dim", lambda: 0)() == 5:
                    B, M, G, H, D = q.shape
                    q = q.reshape(B, M, G * H, D)
                    k = k.reshape(B, k.shape[1], -1, D)
                    v = v.reshape(B, v.shape[1], -1, D)
                    restore = (B, M, G, H, D)
                out = inner(q, k, v, attn_bias=attn_bias, p=p, *args, **kwargs)
                return out.reshape(restore) if restore is not None else out

            wrapped._lab22_t4 = True
            wrapped._lab22_orig = inner
            return wrapped

        for name, attr in (
            ("xformers.ops", "memory_efficient_attention"),
            ("xformers.ops.fmha", "memory_efficient_attention"),
            ("unsloth.models._utils", "xformers_attention"),
            ("unsloth.models.llama", "xformers_attention"),
            ("unsloth.utils.attention_dispatch", "xformers_attention"),
        ):
            try:
                mod = importlib.import_module(name)
                fn = getattr(mod, attr, None)
                if callable(fn):
                    setattr(mod, attr, _wrap(fn))
            except Exception:
                pass
        print("T4 attention patch: xformers GQA 5D→4D (kept memory-efficient kernels)")
        if hasattr(model, "config"):
            model.config.use_cache = False

    import functools
    import torch.utils.checkpoint as tuc
    import transformers.modeling_layers as ml
    _gc = functools.partial(tuc.checkpoint, use_reentrant=False)
    ml.GradientCheckpointingLayer._gradient_checkpointing_func = _gc
    orig_call = ml.GradientCheckpointingLayer.__call__
    if not getattr(orig_call, "_lab22_gc", False):
        def _call(self, *args, **kwargs):
            if "_gradient_checkpointing_func" not in getattr(self, "__dict__", {}):
                object.__setattr__(self, "_gradient_checkpointing_func", _gc)
            return orig_call(self, *args, **kwargs)
        _call._lab22_gc = True
        ml.GradientCheckpointingLayer.__call__ = _call

    train_result = trainer.train()

    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # Headline metrics
    import pandas as pd

    logs = pd.DataFrame(trainer.state.log_history)
    chosen_col = "rewards/chosen" if "rewards/chosen" in logs.columns else None
    rejected_col = "rewards/rejected" if "rewards/rejected" in logs.columns else None

    metrics = {
        "compute_tier": tier,
        "base_model": base_model,
        "beta": args.beta,
        "lr": args.lr,
        "epochs": args.epochs,
        "final_train_loss": float(train_result.training_loss),
        "end_chosen_reward": float(logs[chosen_col].iloc[-5:].mean()) if chosen_col else None,
        "end_rejected_reward": float(logs[rejected_col].iloc[-5:].mean()) if rejected_col else None,
    }
    if metrics["end_chosen_reward"] is not None and metrics["end_rejected_reward"] is not None:
        metrics["end_reward_gap"] = metrics["end_chosen_reward"] - metrics["end_rejected_reward"]

    (output / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nFinal loss:     {train_result.training_loss:.4f}")
    if "end_reward_gap" in metrics:
        print(f"End reward gap: {metrics['end_reward_gap']:+.3f}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
