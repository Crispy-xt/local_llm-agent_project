from tiny_lm.tokenizer import CharacterTokenizer


def test_encode_decode_round_trip() -> None:
    tokenizer = CharacterTokenizer.from_text("ab cab")
    text = "ab cab"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size == 5


def test_unknown_character_uses_unknown_id() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    assert tokenizer.encode("z") == [tokenizer.unk_id]
    assert tokenizer.decode([tokenizer.unk_id]) == tokenizer.unk_token


def test_tokenizer_can_be_saved_and_loaded(tmp_path) -> None:
    tokenizer = CharacterTokenizer.from_text("你好，LLM")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path)
    assert loaded.to_dict() == tokenizer.to_dict()
    assert loaded.decode(loaded.encode("你好，LLM")) == "你好，LLM"