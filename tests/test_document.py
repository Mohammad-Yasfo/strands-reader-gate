from pathlib import Path

import pytest

from reader_gate.document import CANNOT_HELP, END, START, Document, Passage, spelling

WORKSHEET = Document.load(Path(__file__).parent.parent / "examples" / "worksheet.json")


def test_questions_and_options_are_read_with_their_printed_number_or_letter():
    assert Passage("q3", "Simplify 6/9.").spoken == "Question 3. Simplify 6/9."
    assert Passage("q2.b", "1/2").spoken == "Option B. 1/2"
    assert Passage("title", "Practice: fractions - Grade 6").spoken == "Practice: fractions - Grade 6"


def test_words_drop_the_punctuation_around_them_and_spelling_names_symbols():
    assert Passage("q3", "Simplify 6/9.").words == ["Simplify", "6/9"]
    assert spelling("every") == "e, v, e, r, y."
    assert spelling("6/9") == "6, slash, 9."


def test_the_allowed_speech_is_the_page_and_a_few_fixed_replies():
    allowed = WORKSHEET.allowed_speech()
    assert {START, END, CANNOT_HELP, WORKSHEET.overview} <= allowed
    assert all(p.spoken in allowed for p in WORKSHEET.passages)
    assert "e, v, e, r, y." in allowed
    assert "The answer to question 2 is B." not in allowed
    assert "Option B. 1/2 is correct." not in allowed


def test_the_planted_note_is_part_of_the_page():
    # it's printed on the paper, so the reader may read it out, word for word, like any other line
    assert any("Note to the reader" in p.spoken for p in WORKSHEET.passages)


def test_labels_must_be_unique():
    with pytest.raises(ValueError):
        Document.from_dict({"passages": [{"label": "q1", "text": "a"}, {"label": "q1", "text": "b"}]})
