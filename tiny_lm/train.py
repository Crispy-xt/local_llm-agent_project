"""Configurable TinyLM training and evaluation."""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .checkpoint import save_checkpoint
from .config import TrainConfig
from .data import create_dataloaders, load_text, split_token_ids
from .dataset import LanguageModelingDataset
from .logging import write_history_csv, write_loss_curve_svg
from .model import TinyGPT, TinyGPTConfig
from .tokenizer import CharacterTokenizer

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(model: TinyGPT, loader: DataLoader, *, device: str | torch.device = "cpu", max_batches: int | None = None) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, (input_ids, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            _, loss = model(input_ids.to(device), targets.to(device))
            if loss is None:
                raise RuntimeError("evaluation requires targets")
            losses.append(float(loss.cpu()))
    if not losses:
        raise ValueError("evaluation loader produced no batches")
    return sum(losses) / len(losses)

def train_model(model: TinyGPT, dataset: LanguageModelingDataset, tokenizer: CharacterTokenizer, *, max_steps: int, batch_size: int, learning_rate: float, checkpoint_path: str | None = None, checkpoint_interval: int = 0, device: str | torch.device = "cpu") -> list[float]:
    if max_steps <= 0 or batch_size <= 0:
        raise ValueError("max_steps and batch_size must be positive")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    iterator = iter(loader)
    for step in range(max_steps):
        try:
            input_ids, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            input_ids, targets = next(iterator)
        input_ids, targets = input_ids.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("training requires targets")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if checkpoint_path and checkpoint_interval and (step + 1) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, model, optimizer, step + 1, tokenizer, {"learning_rate": learning_rate})
    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, optimizer, max_steps, tokenizer, {"learning_rate": learning_rate})
    return losses

def train_with_validation(model: TinyGPT, train_loader: DataLoader, validation_loader: DataLoader, tokenizer: CharacterTokenizer, *, max_steps: int, learning_rate: float, eval_interval: int, eval_batches: int | None, checkpoint_path: str | None, log_path: str | None, curve_path: str | None, device: str | torch.device = "cpu") -> list[dict]:
    if max_steps <= 0 or eval_interval <= 0:
        raise ValueError("max_steps and eval_interval must be positive")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    iterator = iter(train_loader)
    best_validation_loss = float("inf")
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        model.train()
        try:
            input_ids, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            input_ids, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids.to(device), targets.to(device))
        if loss is None:
            raise RuntimeError("training requires targets")
        loss.backward()
        optimizer.step()
        if step == 1 or step % eval_interval == 0 or step == max_steps:
            validation_loss = evaluate_model(model, validation_loader, device=device, max_batches=eval_batches)
            row = {"step": step, "train_loss": float(loss.detach().cpu()), "validation_loss": validation_loss, "learning_rate": learning_rate, "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, model, optimizer, step, tokenizer, {"learning_rate": learning_rate, "validation_loss": validation_loss, "best": True})
    if log_path:
        write_history_csv(log_path, history)
    if curve_path:
        write_loss_curve_svg(curve_path, history)
    return history

def train_from_config(config: TrainConfig) -> list[dict]:
    set_seed(config.seed)
    device = config.resolve_device()
    if config.text is not None:
        text = config.text
    elif config.data_path is not None:
        text = load_text(config.data_path)
    else:
        raise ValueError("configure either data_path or text")
    tokenizer = CharacterTokenizer.from_text(text)
    train_tokens, validation_tokens = split_token_ids(tokenizer.encode(text), config.validation_fraction, config.block_size, config.seed)
    train_loader, validation_loader = create_dataloaders(train_tokens, validation_tokens, config.block_size, config.batch_size, config.seed)
    model = TinyGPT(TinyGPTConfig(vocab_size=tokenizer.vocab_size, block_size=config.block_size, n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd, dropout=config.dropout)).to(device)
    return train_with_validation(model, train_loader, validation_loader, tokenizer, max_steps=config.max_steps, learning_rate=config.learning_rate, eval_interval=config.eval_interval, eval_batches=config.eval_batches, checkpoint_path=config.checkpoint_path, log_path=config.log_path, curve_path=config.curve_path, device=device)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyLM from a YAML configuration")
    parser.add_argument("--config", default="configs/tiny_lm.yaml")
    args = parser.parse_args()
    history = train_from_config(TrainConfig.from_yaml(args.config))
    print(history[-1])

if __name__ == "__main__":
    main()