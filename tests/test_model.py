import torch

from tiny_lm.checkpoint import load_checkpoint, save_checkpoint
from tiny_lm.dataset import LanguageModelingDataset
from tiny_lm.model import TinyGPT, TinyGPTConfig
from tiny_lm.tokenizer import CharacterTokenizer
from tiny_lm.train import train_model


def build_model(vocab_size: int) -> TinyGPT:
    return TinyGPT(TinyGPTConfig(vocab_size=vocab_size, block_size=8, n_layer=2, n_head=2, n_embd=32))


def test_model_output_shape_and_loss() -> None:
    model = build_model(5)
    input_ids = torch.randint(0, 5, (2, 8))
    targets = torch.randint(0, 5, (2, 8))
    logits, loss = model(input_ids, targets)
    assert logits.shape == (2, 8, 5)
    assert loss is not None and loss.ndim == 0


def test_model_can_generate_tokens() -> None:
    model = build_model(5)
    generated = model.generate(torch.zeros((1, 2), dtype=torch.long), max_new_tokens=3)
    assert generated.shape == (1, 5)


def test_training_loss_decreases_on_repeated_text() -> None:
    torch.manual_seed(0)
    tokenizer = CharacterTokenizer.from_text("abcabcabcabcabcabc")
    dataset = LanguageModelingDataset(tokenizer.encode("abcabcabcabcabcabc"), block_size=6)
    model = build_model(tokenizer.vocab_size)
    losses = train_model(model, dataset, tokenizer, max_steps=60, batch_size=4, learning_rate=0.01)
    assert losses[-1] < losses[0]


def test_checkpoint_restores_model_and_tokenizer(tmp_path) -> None:
    tokenizer = CharacterTokenizer.from_text("abcabc")
    model = build_model(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=7, tokenizer=tokenizer)
    restored, _, restored_tokenizer, step = load_checkpoint(path)
    assert step == 7
    assert restored.config.to_dict() == model.config.to_dict()
    assert restored_tokenizer.to_dict() == tokenizer.to_dict()
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)