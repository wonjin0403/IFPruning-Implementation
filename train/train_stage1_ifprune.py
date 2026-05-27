from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collators import ChunkPairCollator
from models.masked_model import masked_from_pretrained
from models.sparsity_predictor import build_predictor_from_extractor


def save_checkpoint(out_dir: str, llama, predictor, accelerator: Accelerator, step: int):
    if not accelerator.is_main_process:
        return
    ckpt = Path(out_dir) / f"step_{step:08d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    unwrapped_llama = accelerator.unwrap_model(llama)
    unwrapped_pred = accelerator.unwrap_model(predictor)
    unwrapped_llama.save_pretrained(ckpt / "llama", safe_serialization=True)
    torch.save(unwrapped_pred.extractor.state_dict(), ckpt / "extractor.pt")
    torch.save(unwrapped_pred.mask_head.state_dict(), ckpt / "mask_head.pt")
    with open(ckpt / "predictor_meta.json", "w") as f:
        json.dump({
            "num_layers": unwrapped_pred.num_layers,
            "intermediate_size": unwrapped_pred.intermediate_size,
            "topk": unwrapped_pred.topk,
            "mask_method": unwrapped_pred.mask_method,
            "head_hidden_dim": unwrapped_pred.mask_head[0].out_features,
            "extractor_config": unwrapped_pred.extractor.config.to_dict(),
        }, f, indent=2)


@hydra.main(version_base=None, config_path="../configs", config_name="stage1_continued_pretrain")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    log_with = None if cfg.report_to in (None, "none", "no") else cfg.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=log_with,
    )
    set_seed(cfg.seed)

    if accelerator.is_main_process and log_with is not None:
        accelerator.init_trackers(cfg.run_name)

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    predictor_tokenizer_name = cfg.get("predictor_tokenizer", None) or cfg.tokenizer
    pred_tok = AutoTokenizer.from_pretrained(predictor_tokenizer_name, use_fast=True)
    if pred_tok.pad_token_id is None:
        pred_tok.pad_token = pred_tok.eos_token

    init_from = cfg.get("init_from", None)
    if init_from:
        from train.train_stage2_sft import load_stage1_checkpoint
        accelerator.print(f"resuming Llama + predictor from {init_from} ...")
        llama, predictor = load_stage1_checkpoint(init_from, torch_dtype=torch.bfloat16)
    else:
        accelerator.print(f"loading source LLM {cfg.source_model} (mask-aware) ...")
        llama = masked_from_pretrained(
            cfg.source_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=cfg.attn_implementation,
        )
        accelerator.print(f"loading extractor from {cfg.extractor_init} ...")
        predictor = build_predictor_from_extractor(
            cfg.extractor_init,
            num_layers=llama.config.num_hidden_layers,
            intermediate_size=llama.config.intermediate_size,
            topk=cfg.pruning.topk_per_layer,
            head_hidden_dim=cfg.predictor.head_hidden_dim,
            mask_method=cfg.pruning.mask_method,
            torch_dtype=torch.bfloat16,
        )

    gc_kwargs = {"use_reentrant": False}
    if cfg.gradient_checkpointing and cfg.lr_llama > 0:
        llama.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
    if cfg.gradient_checkpointing and not cfg.get("freeze_extractor", False):
        predictor.extractor.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

    train_ds = load_from_disk(cfg.train_dataset)
    collator = ChunkPairCollator(
        pad_token_id=tok.pad_token_id,
        predictor_pad_token_id=pred_tok.pad_token_id,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    freeze_llama = cfg.lr_llama <= 0
    if freeze_llama:
        accelerator.print("Llama frozen (lr_llama <= 0).")
        for p in llama.parameters():
            p.requires_grad_(False)

    freeze_extractor = cfg.get("freeze_extractor", False)
    if freeze_extractor:
        accelerator.print("Predictor extractor frozen; only mask_head is trainable.")
        for p in predictor.extractor.parameters():
            p.requires_grad_(False)

    param_groups = [{"params": [p for p in predictor.parameters() if p.requires_grad], "lr": cfg.lr_predictor}]
    if not freeze_llama:
        param_groups.insert(0, {"params": [p for p in llama.parameters() if p.requires_grad], "lr": cfg.lr_llama})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    num_update_steps = cfg.max_steps
    scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(num_update_steps * cfg.warmup_ratio),
        num_training_steps=num_update_steps,
    )

    llama, predictor, optimizer, train_loader, scheduler = accelerator.prepare(
        llama, predictor, optimizer, train_loader, scheduler
    )

    global_step = 0
    last_log_t = time.time()
    pbar = tqdm(total=num_update_steps, disable=not accelerator.is_main_process, desc="stage1")
    llama.train()
    predictor.train()

    metrics_path = Path(cfg.output_dir) / "metrics.jsonl"
    if accelerator.is_main_process:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _grad_norm(params):
        total_sq = 0.0
        for p in params:
            if p.grad is not None:
                total_sq += float(p.grad.detach().float().pow(2).sum())
        return total_sq ** 0.5

    while global_step < num_update_steps:
        for batch in train_loader:
            with accelerator.accumulate(llama):
                mask, scores = predictor(
                    input_ids=batch["predictor_input_ids"],
                    attention_mask=batch["predictor_attention_mask"],
                )
                outputs = llama(
                    input_ids=batch["llm_input_ids"],
                    attention_mask=batch["llm_attention_mask"],
                    labels=batch["labels"],
                    ffn_mask=mask,
                )
                loss = outputs.loss
                accelerator.backward(loss)

                gn_llama = None
                gn_predictor = None
                gn_extractor = None
                gn_mask_head = None
                ext_count = 0
                head_count = 0
                if accelerator.sync_gradients:
                    if not freeze_llama:
                        gn_llama = _grad_norm(llama.parameters())
                    gn_predictor = _grad_norm(predictor.parameters())
                    ext_params = [p for n, p in predictor.named_parameters() if "extractor" in n]
                    head_params = [p for n, p in predictor.named_parameters() if "mask_head" in n]
                    ext_count = len(ext_params)
                    head_count = len(head_params)
                    gn_extractor = _grad_norm(ext_params)
                    gn_mask_head = _grad_norm(head_params)

                if accelerator.sync_gradients and cfg.max_grad_norm:
                    clip_params = list(predictor.parameters())
                    if not freeze_llama:
                        clip_params = list(llama.parameters()) + clip_params
                    accelerator.clip_grad_norm_(clip_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                pbar.update(1)
                if global_step % cfg.logging_steps == 0:
                    dt = time.time() - last_log_t
                    last_log_t = time.time()
                    with torch.no_grad():
                        sel_freq = mask.float().mean(dim=0).mean(dim=0)
                        sel_entropy = -(sel_freq.clamp_min(1e-8) * sel_freq.clamp_min(1e-8).log()).sum().item()
                        mask_mean_nonzero = mask[mask != 0].float().mean().item() if (mask != 0).any() else 0.0
                    lrs = scheduler.get_last_lr()
                    cur_loss = loss.detach().float().item()
                    metrics = {
                        "step": global_step,
                        "loss": cur_loss,
                        "ppl": math.exp(min(20.0, cur_loss)),
                        "lr_predictor": lrs[-1],
                        "sel_entropy": sel_entropy,
                        "mask_mean_nonzero": mask_mean_nonzero,
                        "step_time_s": dt / cfg.logging_steps,
                        "grad_norm_predictor": gn_predictor,
                        "grad_norm_extractor": gn_extractor,
                        "grad_norm_mask_head": gn_mask_head,
                        "ext_param_count": ext_count,
                        "head_param_count": head_count,
                    }
                    if not freeze_llama:
                        metrics["lr_llama"] = lrs[0]
                        metrics["grad_norm_llama"] = gn_llama
                    accelerator.log({f"train/{k}": v for k, v in metrics.items() if k != "step"}, step=global_step)
                    if accelerator.is_main_process:
                        pbar.set_postfix(loss=f"{cur_loss:.3f}")
                        with open(metrics_path, "a") as f:
                            f.write(json.dumps(metrics) + "\n")
                if global_step % cfg.save_steps == 0:
                    save_checkpoint(cfg.output_dir, llama, predictor, accelerator, global_step)
                if global_step >= num_update_steps:
                    break

    save_checkpoint(cfg.output_dir, llama, predictor, accelerator, global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
