from src.examples import EXAMPLES


def test_examples_have_required_keys():
    for ex in EXAMPLES:
        assert "focus_word" in ex
        assert "sentence_a" in ex
        assert "sentence_b" in ex
        assert ex["focus_word"].lower() in ex["sentence_a"].lower()
        assert ex["focus_word"].lower() in ex["sentence_b"].lower()


def test_examples_count():
    assert len(EXAMPLES) >= 5
