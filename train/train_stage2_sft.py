"""Stage 2: IFPruning SFT.

Same dual-model loop as Stage 1, but:
  - data is Tulu-v2 + FLAN-V2 instead of SlimPajama chunk pairs
  - predictor input is the first user message, llm input is the full chat template
  - labels are -100 outside of assistant tokens (already in the dataset)

Init: load `cfg.init_from` (a Stage 1 checkpoint dir produced by stage1's
save_checkpoint).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import concatenate_datasets, load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, get_scheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collators import SFTCollator
from models.masked_model import masked_class_for
from models.sparsity_predictor import LlamaExtractorSparsityPredictor


def load_stage1_checkpoint(ckpt_dir: str, torch_dtype=torch.bfloat16):
    """Reconstruct (masked_llm, predictor) from a Stage 1 checkpoint dir.

    Works for any supported source LLM (Llama, Qwen2, ...) — dispatched on the
    saved llama/ directory's `config.json:model_type`.
    """
    ckpt = Path(ckpt_dir)
    src_cfg = AutoConfig.from_pretrained(ckpt / "llama")
    masked_cls = masked_class_for(src_cfg.model_type)
    llama = masked_cls(src_cfg).to(dtype=torch_dtype)
    state = AutoModelForCausalLM.from_pretrained(ckpt / "llama", torch_dtype=torch_dtype).state_dict()
    llama.load_state_dict(state, strict=True)

    with open(ckpt / "predictor_meta.json") as f:
        meta = json.load(f)
    ext_cfg = LlamaConfig(**meta["extractor_config"])
    body = AutoModelForCausalLM.from_config(ext_cfg).model.to(dtype=torch_dtype)
    body.load_state_dict(torch.load(ckpt / "extractor.pt", map_location="cpu"), strict=True)
    predictor = LlamaExtractorSparsityPredictor(
        extractor_body=body,
        num_layers=meta["num_layers"],
        intermediate_size=meta["intermediate_size"],
        topk=meta["topk"],
        head_hidden_dim=meta["head_hidden_dim"],
        mask_method=meta["mask_method"],
    ).to(dtype=torch_dtype)
    predictor.mask_head.load_state_dict(torch.load(ckpt / "mask_head.pt", map_location="cpu"), strict=True)
    return llama, predictor


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


@hydra.main(version_base=None, config_path="../configs", config_name="stage2_sft")
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

    accelerator.print(f"loading Stage 1 checkpoint from {cfg.init_from} ...")
    llama, predictor = load_stage1_checkpoint(cfg.init_from)
    if cfg.gradient_checkpointing:
        llama.gradient_checkpointing_enable()
        predictor.extractor.gradient_checkpointing_enable()

    parts = [load_from_disk(p) for p in cfg.train_datasets]
    train_ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    collator = SFTCollator(
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

    params = [
        {"params": llama.parameters(), "lr": cfg.lr_llama},
        {"params": predictor.parameters(), "lr": cfg.lr_predictor},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
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
    pbar = tqdm(total=num_update_steps, disable=not accelerator.is_main_process, desc="stage2")
    llama.train()
    predictor.train()
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
                if accelerator.sync_gradients and cfg.max_grad_norm:
                    accelerator.clip_grad_norm_(
                        list(llama.parameters()) + list(predictor.parameters()),
                        cfg.max_grad_norm,
                    )
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
                    metrics = {
                        "train/loss": loss.detach().float().item(),
                        "train/ppl": math.exp(min(20.0, loss.detach().float().item())),
                        "train/lr_llama": scheduler.get_last_lr()[0],
                        "train/lr_predictor": scheduler.get_last_lr()[1],
                        "train/sel_entropy": sel_entropy,
                        "train/step_time_s": dt / cfg.logging_steps,
                    }
                    accelerator.log(metrics, step=global_step)
                if global_step % cfg.save_steps == 0:
                    save_checkpoint(cfg.output_dir, llama, predictor, accelerator, global_step)
                if global_step >= num_update_steps:
                    break

    save_checkpoint(cfg.output_dir, llama, predictor, accelerator, global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
