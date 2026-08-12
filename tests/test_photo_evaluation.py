from research.scripts.evaluate_photo_filters import _selection_heuristics


def _variant(word_f1: float, numeric_f1: float, words: int) -> dict[str, float | int]:
    return {
        "word_f1": word_f1,
        "numeric_f1": numeric_f1,
        "recognized_words": words,
        "characters": words * 5,
        "model_confidence": word_f1 * 100,
    }


def test_more_words_selects_better_unwarp_and_keeps_raw_on_tie() -> None:
    documents = [
        {
            "variants": {
                "raw": _variant(0.7, 0.6, 7),
                "paddle_unwarp": _variant(0.9, 0.8, 9),
            }
        },
        {
            "variants": {
                "raw": _variant(0.8, 0.9, 8),
                "paddle_unwarp": _variant(0.7, 0.6, 8),
            }
        },
    ]

    result = _selection_heuristics(documents)["more_words"]

    assert result == {
        "mean_word_f1": 0.85,
        "mean_numeric_f1": 0.85,
        "unwarp_picks": 1,
        "oracle_matches": 2,
        "documents": 2,
    }
