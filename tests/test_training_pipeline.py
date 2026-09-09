import csv
import torch

from tiny_lm.config import TrainConfig
from tiny_lm.data import create_dataloaders, split_token_ids
from tiny_lm.logging import write_history_csv, write_loss_curve_svg
from tiny_lm.model import TinyGPT, TinyGPTConfig
from tiny_lm.tokenizer import CharacterTokenizer
from tiny_lm.train import evaluate_model, train_with_validation


def test_split_and_dataloaders_preserve_next_token_shapes() -> None:
    train_tokens, validation_tokens = split_token_ids(list(range(100)), 0.2, block_size=8, seed=42)
    train_loader, validation_loader = create_dataloaders(train_tokens, validation_tokens, 8, 4)
    input_ids, targets = next(iter(train_loader))
    assert len(train_tokens) == 80
    assert len(validation_tokens) == 20
    assert input_ids.shape == (4, 8)
    assert torch.equal(targets, input_ids + 1)
    assert len(validation_loader) > 0


def test_validation_evaluation_returns_scalar() -> None:
    tokenizer = CharacterTokenizer.from_text("abc" * 40)
    tokens = torch.tensor(tokenizer.encode("abc" * 40))
    train_loader, validation_loader = create_dataloaders(tokens[:80], tokens[80:], 8, 4)
    model = TinyGPT(TinyGPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_layer=1, n_head=1, n_embd=16))
    loss = evaluate_model(model, validation_loader)
    assert isinstance(loss, float)
    assert loss > 0


def test_history_writers_create_csv_and_svg(tmp_path) -> None:
    history = [{"step": 1, "train_loss": 1.0, "validation_loss": 1.1, "learning_rate": 0.01, "elapsed_seconds": 0.1}, {"step": 2, "train_loss": 0.5, "validation_loss": 0.6, "learning_rate": 0.01, "elapsed_seconds": 0.2}]
    csv_path = tmp_path / "loss.csv"
    svg_path = tmp_path / "loss.svg"
    write_history_csv(csv_path, history)
    write_loss_curve_svg(svg_path, history)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert "polyline" in svg_path.read_text(encoding="utf-8")


def test_config_driven_training_writes_artifacts(tmp_path) -> None:
    tokenizer = CharacterTokenizer.from_text("abc" * 80)
    tokens = torch.tensor(tokenizer.encode("abc" * 80))
    train_loader, validation_loader = create_dataloaders(tokens[:180], tokens[180:], 8, 4)
    model = TinyGPT(TinyGPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_layer=1, n_head=1, n_embd=16))
    history = train_with_validation(model, train_loader, validation_loader, tokenizer, max_steps=4, learning_rate=0.01, eval_interval=2, eval_batches=2, checkpoint_path=str(tmp_path / "model.pt"), log_path=str(tmp_path / "loss.csv"), curve_path=str(tmp_path / "loss.svg"))
    assert len(history) == 3
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "loss.csv").exists()
    assert (tmp_path / "loss.svg").exists()


def test_yaml_config_loads(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("block_size: 16\nmax_steps: 3\n", encoding="utf-8")
    config = TrainConfig.from_yaml(path)
    assert config.block_size == 16
    assert config.max_steps == 3