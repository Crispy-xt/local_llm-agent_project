"""Text generation helpers and command-line entry point."""

from __future__ import annotations

import argparse

import torch

from .checkpoint import load_checkpoint


def generate_text(checkpoint_path: str, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8) -> str:
    model, _, tokenizer, _ = load_checkpoint(checkpoint_path)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(generated_ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a TinyGPT checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    print(generate_text(args.checkpoint, args.prompt, args.max_new_tokens, args.temperature))


if __name__ == "__main__":
    main()