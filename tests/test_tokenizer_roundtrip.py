from saddlellm.models.TokenizerLoader import load_tokenizer_compatible
from saddlellm.models.TokenizerTrainer import TokenizerTrainer


def test_trained_tokenizer_preserves_special_tokens_after_reload(tmp_path):
    corpus = tmp_path / "tiny-corpus.txt"
    corpus.write_text(
        "A small language model learns from text.\n" * 100,
        encoding="utf-8",
    )
    output = tmp_path / "tokenizer"

    trainer = TokenizerTrainer(vocab_size=64, min_frequency=1, byte_level=False)
    trainer.fit([str(corpus)])
    expected = trainer.get_hf_tokenizer()
    trainer.save(str(output))

    reloaded = load_tokenizer_compatible(str(output), local_files_only=True)

    for token_name in ("pad_token", "unk_token", "bos_token", "eos_token"):
        assert getattr(reloaded, token_name) == getattr(expected, token_name)
        assert getattr(reloaded, f"{token_name}_id") == getattr(
            expected, f"{token_name}_id"
        )


def test_evaluation_ignores_runtime_truncation_and_special_tokens(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta gamma delta epsilon\n" * 20, encoding="utf-8")

    trainer = TokenizerTrainer(
        vocab_size=64,
        min_frequency=1,
        max_token_length=4,
        byte_level=False,
    )
    trainer.fit(str(corpus))
    text = "alpha beta gamma delta epsilon"
    result = trainer.evaluate(texts=[text], domain_terms=["alpha"])

    assert result.roundtrip_exact_rate == 1.0
    assert result.tokens > 4
    assert result.avg_tokens_per_domain_term == 1.0
    assert result.single_token_domain_term_rate == 1.0
    assert trainer._tokenizer.truncation["max_length"] == 4
