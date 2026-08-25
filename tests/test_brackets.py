from app.brackets import age_bracket_from_years, gender_from_probs


def test_gender_picks_argmax():
    assert gender_from_probs({"female": 0.1, "male": 0.9})[0] == "male"
    assert gender_from_probs({"female": 0.8, "male": 0.2})[0] == "female"


def test_gender_child_winner_is_unknown():
    label, _ = gender_from_probs({"female": 0.2, "male": 0.2, "child": 0.6})
    assert label == "unknown"


def test_gender_confidence_is_winning_probability():
    _, conf = gender_from_probs({"female": 0.1, "male": 0.9})
    assert conf == 0.9


def test_age_brackets_map_correctly():
    assert age_bracket_from_years(25)[0] == "18-30"
    assert age_bracket_from_years(38)[0] == "31-45"
    assert age_bracket_from_years(50)[0] == "46-60"
    assert age_bracket_from_years(72)[0] == "60+"


def test_age_under_18_is_unknown():
    assert age_bracket_from_years(10)[0] == "unknown"


def test_age_confidence_higher_at_center_than_edge():
    _, center = age_bracket_from_years(38)   # middle of 31-45
    _, edge = age_bracket_from_years(31)      # on the 31-45 boundary
    assert center > edge
