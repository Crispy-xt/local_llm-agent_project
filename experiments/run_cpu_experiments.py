"""Reproducible CPU experiments for the first TinyLM milestone."""
from __future__ import annotations
import json
import time
from pathlib import Path
import torch
from tiny_lm.dataset import LanguageModelingDataset
from tiny_lm.model import TinyGPT, TinyGPTConfig
from tiny_lm.tokenizer import CharacterTokenizer
from tiny_lm.train import set_seed, train_model
TEXT = "abcabcabcabcabcabcabcabc"
def run_experiment(name: str, learning_rate: float, n_layer: int = 2, n_embd: int = 32) -> dict:
    set_seed(42)
    tokenizer = CharacterTokenizer.from_text(TEXT)
    dataset = LanguageModelingDataset(tokenizer.encode(TEXT), block_size=8)
    model = TinyGPT(TinyGPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_layer=n_layer, n_head=2, n_embd=n_embd, dropout=0.0))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_path = Path("checkpoints") / f"{name}.pt"
    started = time.perf_counter()
    losses = train_model(model, dataset, tokenizer, max_steps=200, batch_size=4, learning_rate=learning_rate, checkpoint_path=checkpoint_path)
    elapsed = time.perf_counter() - started
    prompt_ids = torch.tensor([tokenizer.encode("a")], dtype=torch.long)
    generated_ids = model.generate(prompt_ids, max_new_tokens=20, temperature=0.3)
    result = {"name": name, "learning_rate": learning_rate, "n_layer": n_layer, "n_embd": n_embd, "parameter_count": parameter_count, "steps": len(losses), "initial_loss": losses[0], "final_loss": losses[-1], "minimum_loss": min(losses), "elapsed_seconds": elapsed, "generated_text": tokenizer.decode(generated_ids[0].tolist()), "checkpoint": str(checkpoint_path)}
    print(json.dumps(result, ensure_ascii=False))
    return result
def main() -> None:
    results = []
    for learning_rate in (0.001, 0.01, 0.05):
        results.append(run_experiment(f"lr_{str(learning_rate).replace('.', '_')}", learning_rate=learning_rate))
    results.append(run_experiment("small_model", learning_rate=0.01, n_layer=1, n_embd=16))
    results.append(run_experiment("larger_model", learning_rate=0.01, n_layer=3, n_embd=48))
    output_path = Path("experiments") / "cpu_tinylm_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved results to {output_path}")
if __name__ == "__main__":
    main()